"""Every key the search loop reads off a compile result is one the compiler writes.

The compile result is a plain dict passed across a module boundary, so a
deleted producer key does not fail at import, at call, or in any unit test --
it turns into ``result.get(k) or {}`` and the consumer silently reads an empty
default. That is not a crash, it is a behavioral regression with no signal.

Measured, on this repo: removing the multi-reasoner option deleted the
``ensemble`` block from the compiler's result while ``search_loop`` still read
``result["ensemble"]["alive_rids"]`` to decide which reasoners were alive for
the flight. The set came back empty, so ZERO reasoners were armed, the per-stop
reshuffle never ran, and the commit fell through to the first detection
(``detected_0``). 983 unit tests stayed green and a 43-episode paired fleet
scored 8/43 against the baseline's 15/43 before anything said a word.

A key can also break the other way, without ever going missing: the compiler
declares it and hardcodes it empty, so the read succeeds and the branch behind
it never turns over. ``receipt["clauses"]`` sat like that on every episode ever
flown, gating the path that builds a TargetSpec from a parse.

This is a source scan, not an import test: it reads every ``result.get("k")``
in the consumer and checks ``k`` against the string keys the compiler actually
writes, and against the ones it writes only as ``[]`` / ``{}``.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]

PRODUCER = "maps/search/parse_compile/compiler.py"
CONSUMERS = (
    "maps/search/search_loop.py",
    "benchmarks/citynav/run_benchmark/episode_runner.py",
)

# Consumer variable name -> the producer payload it holds. ``""`` is the
# compile result itself; a named payload is one of its nested blocks.
RESULT_NAMES = {
    "result": "",
    "compile_result": "",
    "pre_raw": "",
    "turn_raw": "",
}

# Where a retired payload name may not reappear.
NAME_SCAN_ROOTS = ("maps", "benchmarks")


def _written_keys(path: pathlib.Path) -> set[str]:
    """Every string key the producer can put on a result dict.

    Deliberately broad: any dict-literal key, any ``d["k"] =`` subscript
    assignment, any ``setdefault("k", ...)``. Over-collecting only weakens the
    test; it never fails a key the producer really writes.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            out |= {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Subscript) and isinstance(tgt.slice, ast.Constant):
                    if isinstance(tgt.slice.value, str):
                        out.add(tgt.slice.value)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            out.add(node.args[0].value)
    return out


def _is_empty_literal(value: ast.AST) -> bool:
    return (
        isinstance(value, (ast.List, ast.Dict, ast.Tuple, ast.Set))
        and not getattr(value, "elts", None)
        and not getattr(value, "keys", None)
    )


def _empty_only_keys(path: pathlib.Path, payload: str) -> set[str]:
    """Keys of ``payload`` the producer writes only ever as ``[]`` / ``{}``.

    A key like this passes the name check above -- the producer really does
    write it -- while carrying nothing on every episode ever run. A consumer
    that branches on it takes the same path forever, which is a dead contract
    wearing a live one's clothes.

    A key is spared as soon as ONE write gives it a non-empty value, so an
    error-path default (``"alive_rids": []`` beside the real
    ``"alive_rids": list(alive_rids)``) is not flagged.

    ``payload=""`` scans the result itself: every dict literal, subscript
    assignment and ``setdefault`` in the producer. A named payload scans only
    the dict literals written as ``"<payload>": {...}``, so the same key name
    on two payloads stays two contracts.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    seen: dict[str, bool] = {}  # key -> every write so far was empty

    def _note(key: str, value: ast.AST) -> None:
        seen[key] = _is_empty_literal(value) and seen.get(key, True)

    def _note_dict(node: ast.Dict) -> None:
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                _note(k.value, v)

    if payload:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and k.value == payload
                        and isinstance(v, ast.Dict)):
                    _note_dict(v)
        return {k for k, always_empty in seen.items() if always_empty}

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            _note_dict(node)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.slice, ast.Constant)
                        and isinstance(tgt.slice.value, str)):
                    _note(tgt.slice.value, node.value)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault" and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            _note(node.args[0].value, node.args[1])
    return {k for k, always_empty in seen.items() if always_empty}


def _read_keys(path: pathlib.Path) -> list[tuple[str, str, int]]:
    """``(payload, key, lineno)`` for every ``<result>.get("key")`` read."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in RESULT_NAMES
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            payload = RESULT_NAMES[node.func.value.id]
            out.append((payload, node.args[0].value, node.lineno))
    return out


def test_no_consumer_reads_a_key_the_compiler_never_writes():
    written = _written_keys(REPO / PRODUCER)
    offenders: list[str] = []
    for rel in CONSUMERS:
        for payload, key, lineno in _read_keys(REPO / rel):
            if key not in written:
                where = f"{payload}[{key!r}]" if payload else f"result[{key!r}]"
                offenders.append(f"{rel}:{lineno} reads {where}")
    assert offenders == [], (
        f"{PRODUCER} never writes these keys, so the read silently returns a "
        f"default instead of failing:\n  " + "\n  ".join(offenders)
    )


def test_no_consumer_reads_a_key_the_producer_always_leaves_empty():
    """Measured, on this repo: the ``receipt`` block declared ``clauses`` and
    ``unresolved_clauses`` and hardcoded both to ``[]`` -- on ``main`` as well,
    so on every episode ever flown. Three packages branched on ``clauses``
    anyway. ``search_loop._build_target_spec`` gated the whole
    receipt-as-parse path behind it, so the gate was false forever and the
    spec was never once built from a parse; the runner's per-turn clause
    histogram printed all-zeros; and the failure-mode classifier's candidate
    union silently lost a source.

    The name check above cannot see this: the producer DOES write the key.
    """
    empty_only = {
        payload: _empty_only_keys(REPO / PRODUCER, payload)
        for payload in set(RESULT_NAMES.values())
    }
    offenders: list[str] = []
    for rel in CONSUMERS:
        for payload, key, lineno in _read_keys(REPO / rel):
            if key in empty_only[payload]:
                where = f"{payload}[{key!r}]" if payload else f"result[{key!r}]"
                offenders.append(f"{rel}:{lineno} reads {where}")
    assert offenders == [], (
        f"{PRODUCER} only ever writes these keys as an empty literal, so the "
        f"read below is a branch that never turns over:\n  "
        + "\n  ".join(offenders)
    )


def test_the_retired_ensemble_block_has_no_readers_left():
    """``ensemble`` was the multi-reasoner payload; the option is gone.

    Pinned by name because its removal is what motivated this file: a stale
    reader of this specific key silently disarmed every reasoner.
    """
    offenders = _name_scan(("ensemble",), exempt=())
    assert offenders == [], (
        "the multi-reasoner ensemble payload is retired; these still name it:\n  "
        + "\n  ".join(offenders)
    )


# The uniform-loop cutover's retired names. Each was a phase-era surface
# (the iteration-0 toolset/terminal, the filed noun, the parse-tree stub);
# a reappearance is either a resurrected alias or a stale reader.
RETIRED_UNIFORM_LOOP_NAMES = (
    "target_noun",
    "parse_tree",
    "ACT_PLAN_TOOL",
    "ITER0_TOOLS",
    "PER_STOP_TOOLS",
    "run_iteration0",
    "act{plan",
)

# Files whose JOB is to name the retirements (this scanner, and the
# round-0 suite's absence checks).
_NAME_SCAN_EXEMPT = ("maps/search/tests/test_round0.py",)


def _name_scan(banned: tuple[str, ...], exempt: tuple[str, ...]) -> list[str]:
    here = pathlib.Path(__file__).resolve()
    offenders: list[str] = []
    n_files = 0
    for root in NAME_SCAN_ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            rel = str(path.relative_to(REPO))
            if (path.resolve() == here or "__pycache__" in str(path)
                    or rel in exempt):
                continue
            n_files += 1
            for i, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if line.lstrip().startswith("#"):
                    continue
                if any(b in line for b in banned):
                    offenders.append(f"{rel}:{i}: {line.strip()}")
    # Vacuity floor: the scan must actually have covered the tree.
    assert n_files > 200, f"name scan only saw {n_files} files"
    return offenders


def test_the_uniform_loop_retirements_do_not_reappear():
    """One toolset, one act family, no filed noun, no parse-tree stub.

    The phase-era names may not come back under ``maps/`` or
    ``benchmarks/`` outside a comment: a live occurrence is either a
    compat alias (banned by the clean-cutover rule) or a reader of a key
    no producer writes.
    """
    offenders = _name_scan(RETIRED_UNIFORM_LOOP_NAMES, _NAME_SCAN_EXEMPT)
    assert offenders == [], (
        "the uniform-loop cutover retired these names; they reappeared:\n  "
        + "\n  ".join(offenders)
    )


# The commit-landing cutover's retired names: the select's basis apparatus
# (justification lives in the think span), the [x,y] world-point select
# form, and the saw:/delivered-crop channel that existed only for the
# basis split. ``"xy_literal"`` is banned as the QUOTED label only;
# ``select_xy`` is NOT banned (substring of the live
# ``rejected_select_xy`` gate-veto channel).
RETIRED_LANDING_CUTOVER_NAMES = (
    "select_basis",
    "_basis_split",
    "SELECT_XY",
    "_xy_literal_of",
    '"xy_literal"',
    "basis_note",
    "delivered_crop_stops",
)


def test_the_landing_cutover_retirements_do_not_reappear():
    """select is {fid} alone; the commit binds the selected fid's
    centroid — never a basis echo, never a coordinate smuggled as a fid."""
    offenders = _name_scan(RETIRED_LANDING_CUTOVER_NAMES, _NAME_SCAN_EXEMPT)
    assert offenders == [], (
        "the commit-landing cutover retired these names; they reappeared:\n  "
        + "\n  ".join(offenders)
    )
