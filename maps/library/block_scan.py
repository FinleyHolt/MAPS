"""What a block may contain: the calls a run binds (less those a withhold
unbinds), the builtins and modules its namespace serves, and the pre-scan
that vets one block before it runs: plain Python, minus non-whitelisted
imports, underscores and moves a loop or a function could repeat."""

from __future__ import annotations

import ast
import bisect
import builtins
import collections
import functools
import heapq
import itertools
import json
import math
import operator
import re

from maps.library.op_compiler import CodeError
from maps.withhold import withheld


BLOCK_FILENAME = "<mission-block>"

_MOVE_CALLS = ("go_to", "explore")
_MOVE_SAYS = {"go_to": "go to", "explore": "cover"}
# order/filter are METHODS on a result (rows.order(...), rows.filter(...)),
# served through Rows.
_LIBRARY_CALLS = ("find", "look_for", "go_to", "explore", "look",
                  "show_map", "stand", "more_of",
                  "intersection", "ends_of", "distance")

# A call a bench withhold (maps.withhold) unbinds -> that withhold.
_WITHHELD_BY = {"look_for": "look_for", "look": "look", "show_map": "map",
                **dict.fromkeys(("find", "explore", "stand", "more_of",
                                 "intersection", "ends_of", "distance"), "library")}
# The rows' methods and tiers the ``library`` withhold hides: an access reads
# as an attribute the rows never had.
_LIBRARY_ATTRS = frozenset({"filter", "order", "distance", "exact", "tagged", "fuzzy"})


def served_calls() -> tuple[str, ...]:
    """The mission calls this run binds: every call less those a withhold
    unbinds."""
    return tuple(c for c in _LIBRARY_CALLS
                 if c not in _WITHHELD_BY or not withheld(_WITHHELD_BY[c]))


# Free-call spellings the surface refuses, mapped to the method that performs
# the operation. The refusal names the method; nothing here defines the free
# name or runs anything.
_MOVED_TO_METHOD = {
    "narrow": "rows.filter(near=<place>, dwithin=<metres>)",
    "order_by": "rows.order(nearest=<place>)",
    # The one distance is the footprint gap, spelled as the served column
    # spells it ("distance to <ref>").
    "gap": "distance(a, b), or a.distance(b)",
}

_FORBIDDEN_NAMES = frozenset({
    "exec", "eval", "compile", "open", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "input", "exit",
    "quit", "help"})

_SAFE_BUILTINS = {n: getattr(builtins, n)
                  for n in ("len", "min", "max", "sorted", "sum", "abs",
                            "round", "range", "enumerate", "zip", "str",
                            "int", "float", "bool", "list", "dict",
                            "tuple", "set", "reversed", "any", "all",
                            "isinstance", "hasattr", "map", "filter",
                            "repr", "divmod", "pow", "type", "dir",
                            "next",
                            # the standard exception names, for
                            # try/except
                            "Exception", "ValueError", "KeyError",
                            "TypeError", "IndexError", "AttributeError",
                            "ZeroDivisionError", "RuntimeError",
                            "StopIteration", "ArithmeticError",
                            "LookupError")}

# Pure, deterministic stdlib only: no I/O, no clock, no randomness --
# anything ambient would break identical bytes -> identical trajectory, and
# numpy-class C loops would bypass the trace-line liveness ceiling.
_SAFE_MODULES = {m.__name__: m for m in (
    bisect, collections, functools, heapq, itertools, json, math,
    operator, re)}


def _import_refusal(mod: str) -> str:
    return (f"cannot import {mod}; the modules already loaded -- "
            + ", ".join(sorted(_SAFE_MODULES)) + " -- need no import")


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """The one importer the namespace serves: the whitelist, nothing else.
    (The import STATEMENT resolves through ``__builtins__['__import__']``,
    so the pre-scan's refusal needs this runtime twin.)"""
    if level or name.split(".")[0] not in _SAFE_MODULES:
        raise CodeError(_import_refusal(name.split(".")[0] or "that"))
    return __import__(name, globals, locals, fromlist, level)


_LOOP_MOVE_MSG = (
    "{name}() cannot go inside a loop or a def/lambda: it MOVES, and a loop moves to "
    "every row of the set{last}{explore}. "
    "{then}, then move once to the place you chose.")


def _shadows(name: str) -> str:
    """The remedy for a mission call's name the model bound over."""
    return f"binding {name} shadows the {name}() call; del {name} gives the call back."


def scan_block(code: str, bound=()) -> ast.Module:
    """Parse and vet one block, or raise :class:`CodeError` with the line
    and the reason (a syntax error echoes the offending source line with a
    caret); ``bound`` holds the names earlier blocks bound."""
    try:
        tree = ast.parse(code)
        # Some syntax errors (a repeated keyword argument) are raised by the
        # compiler, not the parser.
        compile(tree, BLOCK_FILENAME, "exec")
    except SyntaxError as e:
        where = ""
        if e.text and e.text.strip():
            line = e.text.rstrip("\n")
            if "\n" in line:
                # A continuation line spans physical rows; a caret under
                # the wrong row misleads worse than no caret. Echo only.
                where = "\n    " + "\n    ".join(line.splitlines())
            else:
                # The pad is measured on the TAB-EXPANDED prefix so the
                # caret lands under the offending column as rendered.
                cut = max((e.offset or 1) - 1, 0)
                pad = len(line[:cut].expandtabs(4))
                where = (f"\n    {line.expandtabs(4)}\n    "
                         + " " * pad + "^")
        raise CodeError(
            f"line {e.lineno}: syntax error: {e.msg}{where}") from None
    loop_depth = 0
    at = 1                    # the line of the node being vetted
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    # A move's name is the model's variable where a store of it comes first
    # (earlier in this block, or in an earlier block). A comprehension binds
    # its targets over its own span, its element (written first) included.
    first_store: dict = {}
    comp_spans: list = []        # (name, start, end) a comprehension binds
    comp_targets: set = set()
    for n in ast.walk(tree):     # breadth first: a comprehension before its targets
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for g in n.generators:
                for t in ast.walk(g.target):
                    if isinstance(t, ast.Name):
                        comp_targets.add(id(t))
                        comp_spans.append((t.id, (n.lineno, n.col_offset),
                                           (n.end_lineno, n.end_col_offset)))
        elif isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            # Parameters are bound over the function's own span.
            for arg in ast.walk(n.args):
                if isinstance(arg, ast.arg):
                    comp_spans.append((arg.arg, (n.lineno, n.col_offset),
                                       (n.end_lineno, n.end_col_offset)))
        elif (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
              and id(n) not in comp_targets):
            pos = (n.lineno, n.col_offset)
            first_store[n.id] = min(first_store.get(n.id, pos), pos)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ExceptHandler)) and n.name:
            pos = (n.lineno, n.col_offset)
            first_store[n.name] = min(first_store.get(n.name, pos), pos)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                k = al.asname or al.name.split(".")[0]
                pos = (n.lineno, n.col_offset)
                first_store[k] = min(first_store.get(k, pos), pos)

    def _stored_before(n) -> bool:
        pos = (n.lineno, n.col_offset)
        return (n.id in bound or first_store.get(n.id, (1 << 30, 0)) < pos
                or any(k == n.id and lo <= pos <= hi for k, lo, hi in comp_spans))

    def _walk(node):
        nonlocal loop_depth, at
        at = getattr(node, "lineno", at)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = ([node.module] if isinstance(node, ast.ImportFrom)
                    else [a.name for a in node.names])
            if getattr(node, "level", 0):
                raise CodeError(_import_refusal("a relative module"))
            for m in mods:
                root = (m or "").split(".")[0]
                if root not in _SAFE_MODULES:
                    raise CodeError(_import_refusal(root or "that"))
        if isinstance(node, ast.ClassDef):
            raise CodeError("no class definitions -- write the calls "
                            "you want run")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise CodeError("no global/nonlocal")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise CodeError(f"no underscore attributes (.{node.attr})")
        if isinstance(node, ast.Name):
            if node.id in _MOVE_CALLS and isinstance(node.ctx, ast.Load) \
                    and id(node) not in called and not _stored_before(node) \
                    and not withheld("loop"):
                raise CodeError(
                    f"{node.id} is a move; call it once, {node.id}(<place>), "
                    "outside any loop or function.")
            if node.id.startswith("_"):
                raise CodeError(f"no underscore names ({node.id})")
            if node.id in _FORBIDDEN_NAMES:
                raise CodeError(f"there is no {node.id}()")
            if isinstance(node.ctx, ast.Store) \
                    and node.id in _SAFE_MODULES:
                # one bad bind would break the module for the rest of the
                # mission
                raise CodeError(
                    f"{node.id} is a loaded module -- binding over it "
                    f"would shadow it for the rest of the mission. Pick "
                    "another name.")
        in_loop = isinstance(node, (ast.For, ast.While, ast.AsyncFor, ast.ListComp,
                                    ast.SetComp, ast.DictComp, ast.GeneratorExp,
                                    ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.Lambda))
        if in_loop:
            loop_depth += 1
        if (loop_depth and not withheld("loop") and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _MOVE_CALLS):
            raise CodeError(_LOOP_MOVE_MSG.format(
                name=node.func.id,
                last="" if withheld("end_commit") else " -- go_to commits to whichever came last",
                then="Loop to read the rows"
                if withheld("look") else "Loop to LOOK",
                explore="" if withheld("library") else
                ", and explore flies a covering pass over each"))
        for child in ast.iter_child_nodes(node):
            _walk(child)
        if in_loop:
            loop_depth -= 1

    try:
        _walk(tree)
    except CodeError as e:
        raise CodeError(f"line {at}: {e}") from None
    return tree


def _instrument(tree: ast.Module) -> ast.Module:
    """Wrap top-level bare expressions in a show call, REPL-style
    (``pools`` alone shows the rows, ``len(pools)`` prints the number);
    everything else, control flow included, is the model's."""
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Expr):
            label = ast.unparse(node.value)
            call = ast.Call(
                func=ast.Name(id="__show__", ctx=ast.Load()),
                args=[node.value, ast.Constant(value=label)], keywords=[])
            tree.body[i] = ast.copy_location(
                ast.Expr(value=ast.copy_location(call, node)), node)
    ast.fix_missing_locations(tree)
    return tree


def looks_like_code(text: str) -> bool:
    """A plain reply that is really a block: it parses AND calls a call
    this run binds, which separates code written outside the tool from prose that
    happens to parse."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    calls = served_calls()
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id in calls for n in ast.walk(tree))
