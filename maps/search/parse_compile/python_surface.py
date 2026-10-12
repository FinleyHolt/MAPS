"""Real-Python front-end for the mission run channel.

Every mission call (`find`, `go_to`, `look_for`, ...) is a real function in one
persistent per-mission namespace, so the model's Python behaves as Python and
an error costs only the line it is on. Each call compiles its arguments
(``code_surface``) and runs on the loop's verbs (:class:`MissionEnv`). Moving
calls take a thing, never a coordinate or a whole result set, and a move inside
a loop is refused up front."""

from __future__ import annotations

import ast
import bisect
import builtins
import collections
import contextlib
import difflib
import functools
import heapq
import inspect
import itertools
import json
import math
import operator
import re
import sys
import threading
import time
import traceback
import types
from dataclasses import dataclass
from typing import Callable

from maps.search.analyze_scene.geom_compute import COMPASS_UNIT
from maps.search.analyze_scene.geom_program import (
    _AUTOATTACH_PREFIXES, _BROAD_HEAD_ROWS, _STANDING_SIDE_PREFIXES,
    MODEL_UNSERVED_COLUMNS)
from maps.search.analyze_scene.row_geometry import SurfaceGeometry
from maps.search.withhold import withheld
from maps.util.xy_token import format_xy_token, is_xy, parse_xy_token
from maps.search.parse_compile.code_surface import (
    _CARDINALS, Bind, CodeError, Program, Ref, act_needs_a_thing, bind_row,
    compile_filter, compile_find, compile_order, _fid_like,
    _refuse_metres_in_within, rn_spelled, _split_diagonal)

BLOCK_FILENAME = "<mission-block>"

# Liveness ceiling on one block's own lines (harness frames are untraced).
# A block that runs 100k model-written lines is a runaway loop, not a plan.
TRACE_LINE_CEILING = 100_000

# Wall-clock ceiling on ONE block's execution: it catches a BOUNDED loop
# whose calls are each slow (`for d in rows: find(near=d, dwithin=40)` over
# the live world state), since the runner thread is the one executing the
# block. Past it the block stops at its line with a served error.
TRACE_DEADLINE_S = 90.0

_MOVE_CALLS = ("go_to", "explore")
# order/filter are METHODS on a result (rows.order(...), rows.filter(...)),
# served through Rows.
_LIBRARY_CALLS = ("find", "look_for", "go_to", "explore", "look",
                  "show_map", "stand", "more_of",
                  "intersection", "ends_of", "distance")


def served_calls() -> tuple[str, ...]:
    """The mission calls this run binds: a ``look_for`` or ``look`` withhold
    (maps.search.withhold) unbinds the verb of its name."""
    return tuple(c for c in _LIBRARY_CALLS
                 if c not in ("look_for", "look") or not withheld(c))

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
# `statistics` is out: NormalDist.samples() draws from ambient random.gauss.
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

# The printed channel is capped the way a notebook caps cell output, so a
# loop that prints a city-wide result cannot flood the context.
PRINT_LINE_CAP = 60
PRINT_CHAR_CAP = 8_000

_LOOP_MOVE_MSG = (
    "{name}() cannot go inside a loop: it MOVES, and a loop moves to "
    "every row of the set -- go_to commits to whichever came last, and "
    "explore flies a covering pass over each. Loop to LOOK, then move "
    "once to the place you chose.")


class _Stop(Exception):
    """A plan-level refusal already folded into the receipt: end the block."""


class MissionNamespace(dict):
    """The exec globals (CPython honours ``__missing__`` on a dict-subclass
    globals): an undefined name that spells a served fid is a :class:`Row` of
    the live row, else the fid string; a name this mission unbound says why,
    and a near-miss gets a did-you-mean."""

    mp: "MissionPython | None" = None

    def __missing__(self, k):
        mp = self.mp
        if _fid_like(k):
            row = mp.env.row_of_fid(k) if mp else None
            return Row(row, mp=mp) if isinstance(row, dict) else k
        # A plain KeyError is what lets LOAD_GLOBAL fall through to
        # __builtins__.
        if k in (dict.get(self, "__builtins__") or {}):
            raise KeyError(k)
        if mp and k in mp._unbound:
            raise CodeError(f"{k} is unbound: {mp._unbound[k]}")
        # A near-miss of a LIVE fid names the correction (detect_59 ->
        # detected_59), but ONLY when the prefix is a near-miss of a fid
        # family: a `corner_5` typo falls through to the bound handles.
        m = re.fullmatch(r"([A-Za-z]+)_(\d+)", k)
        if m and mp:
            for pref in ("detected", "prior", "stop"):
                if not difflib.get_close_matches(
                        m.group(1), [pref], n=1, cutoff=0.75):
                    continue
                fix = f"{pref}_{m.group(2)}"
                if fix != k and mp.env.row_of_fid(fix) is not None:
                    raise CodeError(f"there is no {k} on this surface -- "
                                    f"did you mean {fix}?")
        # A free spelling of a method: the name stays undefined and the
        # refusal names the method.
        _moved = difflib.get_close_matches(k, tuple(_MOVED_TO_METHOD),
                                           n=1, cutoff=0.8)
        if k in _MOVED_TO_METHOD or _moved:
            k = k if k in _MOVED_TO_METHOD else _moved[0]
            raise CodeError(
                f"there is no {k}() here -- did you mean "
                f"{_MOVED_TO_METHOD[k]}? A cut and a sort are methods on the "
                "rows they act on.")
        # The suggestion pool: the mission calls and the model's own bound
        # handles, never the module names.
        known = [n for n in self if isinstance(n, str)
                 and not n.startswith("__") and n not in _SAFE_MODULES]
        close = difflib.get_close_matches(
            k, served_calls() + tuple(known), n=2, cutoff=0.6)
        if close:
            raise CodeError(f"there is no {k} here -- did you mean "
                            + " or ".join(dict.fromkeys(close)) + "?")
        raise KeyError(k)


# --------------------------------------------------------------------------
# The pre-scan: plain Python, minus non-whitelisted imports, underscores
# and moving loops
# --------------------------------------------------------------------------


def scan_block(code: str) -> ast.Module:
    """Parse and vet one block. Raises :class:`CodeError` with the reason.

    A syntax error echoes the offending source line with a caret.
    """
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

    def _walk(node):
        nonlocal loop_depth
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
            if node.id.startswith("_"):
                raise CodeError(f"no underscore names ({node.id})")
            if node.id in _FORBIDDEN_NAMES:
                raise CodeError(f"there is no {node.id}()")
            if isinstance(node.ctx, ast.Store) \
                    and node.id in served_calls():
                raise CodeError(
                    f"{node.id} is a mission call -- binding over it "
                    f"would make {node.id}() uncallable for the rest of "
                    "the mission. Pick another name.")
            if isinstance(node.ctx, ast.Store) \
                    and node.id in _SAFE_MODULES:
                # one bad bind would break the module for the rest of the
                # mission
                raise CodeError(
                    f"{node.id} is a loaded module -- binding over it "
                    f"would shadow it for the rest of the mission. Pick "
                    "another name.")
        in_loop = isinstance(node, (ast.For, ast.While, ast.AsyncFor))
        if in_loop:
            loop_depth += 1
        if (loop_depth and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _MOVE_CALLS):
            raise CodeError(_LOOP_MOVE_MSG.format(name=node.func.id))
        for child in ast.iter_child_nodes(node):
            _walk(child)
        if in_loop:
            loop_depth -= 1

    _walk(tree)
    return tree


def _instrument(tree: ast.Module) -> ast.Module:
    """Wrap top-level bare expressions in a show call, REPL-style.

    ``pools`` on its own line shows the rows; ``len(pools)`` on its own
    line prints the number. Everything else is untouched, so control flow
    is exactly the model's.
    """
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
    """A plain reply that is really a block: parses AND calls the library.

    The repair nudge's detector; the library-call requirement separates
    code written outside the tool from prose that happens to parse.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id in _LIBRARY_CALLS for n in ast.walk(tree))


# --------------------------------------------------------------------------
# The values mission calls hand back
# --------------------------------------------------------------------------


def _norm_key(k: str) -> str:
    """A column spelling collapsed the way ``geom_program._sanitize_token``
    collapses ref labels, so "Poppy Boulevard" names the column "facing
    poppy_boulevard"."""
    return re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")


# The cardinal cut kwargs, derived from the compiler's (the geometry's) four.
_CARDINAL_KWARGS = frozenset(f"{w}_of" for w in _CARDINALS)


def _direction_key(word):
    """A compass word -> the sort key that puts the furthest that way first:
    the rows' own ``xy`` (+x east, +y north) along the compass unit vector. A
    direction orders globally and takes no anchor."""
    w = str(word).strip().lower()
    if w not in COMPASS_UNIT:
        raise CodeError(
            f"{word!r} is not a compass word: {'|'.join(COMPASS_UNIT)}. "
            f"To order by a PLACE instead of a direction, that is "
            f"order(nearest=<place>).")
    ux, uy = COMPASS_UNIT[w]

    def _key(r):
        xy = r.xy
        return -(ux * float(xy[0]) + uy * float(xy[1]))

    return _key


def _refuse_reversed_direction(direction, reverse) -> None:
    """``order(direction=..., reverse=True)`` is the OPPOSITE direction,
    so it is refused with the word for it (``reverse=`` is served on a
    key= sort)."""
    if not reverse:
        return
    w = str(direction).strip().lower()
    opp = {"north": "south", "south": "north", "east": "west", "west": "east"}
    flip = "".join(opp.get(p, p) for p in (_split_diagonal(w) or (w,)))
    raise CodeError(
        f'order(direction="{w}", reverse=True) is '
        f'order(direction="{flip}") -- say the direction you mean.')


def _diagonal_remedy(names) -> str:
    """The extra sentence a ``<diag>_of=`` mistake earns: the conjunction
    of its two cardinal cuts against the same place."""
    for n in names:
        n = str(n)
        pair = _split_diagonal(n[:-3]) if n.endswith("_of") else None
        if pair:
            return (f" A diagonal is TWO cuts against the same place: "
                    f"{pair[0]}_of=<place>, {pair[1]}_of=<place> -- which is "
                    f"how you say a corner.")
    return ""


_FILTER_KWARGS = _CARDINAL_KWARGS | {"near", "within", "dwithin", "facing", "direction"}


def _refuse_pred_and_cut(pred, kw: dict) -> None:
    """A predicate and a cut kwarg together are refused, one text for every
    filter."""
    if pred is not None and any(v is not None for v in kw.values()):
        raise CodeError("filter takes a predicate OR near=/within=/dwithin=/"
                        "facing=/east_of=/north_of=/south_of=/west_of=, not both")


def _refuse_filter_kwargs(names) -> None:
    """A filter kwarg outside the set is refused, never ignored: one text
    for every filter, and a colour cut gets its own reason."""
    if "color" in names:
        raise CodeError(
            "colour never CUTS -- it orders. The same car reads red "
            "0.63 through a tight mask and 0.38 through its full one, "
            "so a colour threshold drops the right row on the width of "
            "its mask. Say rows.order(color=\"red\") and let the "
            "spatial cuts do the dropping.")
    bad = sorted(k for k in names if k not in _FILTER_KWARGS)
    if bad:
        raise CodeError(
            f"filter takes near=/within=/dwithin=/facing=, "
            f"{'/'.join(sorted(_CARDINAL_KWARGS))} or a predicate; "
            f"{'=, '.join(bad)}= is none of them."
            + _diagonal_remedy(bad))


_MISSING = object()


class Row:
    """One selected row: a read-only view of its served columns (attributes,
    ``r["spaced name"]``, ``.get``/``.keys``, ``print``), geometry blobs
    hidden. ``bind`` is what a slot takes (the fid, or [x, y] for a computed
    row); ``src`` is the (ref, index) it was indexed from, so showing it
    renders the row."""

    __slots__ = ("bind", "fid", "name", "src", "_row", "_mp")

    def __init__(self, row: dict, src: tuple | None = None, mp=None):
        object.__setattr__(self, "_row", dict(row))
        object.__setattr__(self, "_mp", mp)  # receipt-note sink, optional
        self.bind = bind_row(row)
        self.fid = row.get("fid")
        self.name = row.get("name")
        self.src = src                       # (ref, index) or None

    # ONE row cuts and orders as the one-element set it is:
    # r.order(nearest=poppy) puts a "distance to poppy" column on the row
    # you are about to commit.
    def distance(self, other) -> float:
        """Metres from this row's footprint to ``other``."""
        mp = object.__getattribute__(self, "_mp")
        return mp._call_distance(self, other)

    def filter(self, pred=None, **kw):
        mp = object.__getattribute__(self, "_mp")
        _refuse_filter_kwargs(kw)
        _refuse_pred_and_cut(pred, kw)
        if pred is not None:
            return RowList([self] if pred(self) else [])
        return mp._rows_filter(self, **kw)

    def order(self, key=None, **kw):
        mp = object.__getattribute__(self, "_mp")
        if key is not None or "direction" in kw:
            if kw and key is not None:
                raise CodeError("order takes a key OR "
                                "nearest=/color=/direction=, not both")
            if "direction" in kw:
                # validated, never ignored: one row is already in order,
                # but a bad word must read the same here as anywhere
                _direction_key(kw["direction"])
            return RowList([self])
        return mp._rows_order(self, **kw)

    def _resolve_key(self, k):
        """The served column a Python spelling names, or None: the exact key,
        else the one column equal under :func:`_norm_key` (an ambiguous match
        is None). Attribute, item and ``.get`` access all resolve here."""
        row = object.__getattribute__(self, "_row")
        if not isinstance(k, str) or k in MODEL_UNSERVED_COLUMNS:
            return None
        if k in row:
            return k
        hit = [rk for rk in row
               if rk not in MODEL_UNSERVED_COLUMNS and _norm_key(rk) == _norm_key(k)]
        return hit[0] if len(hit) == 1 else None

    def _derived(self, k):
        """r.x / r.y off the xy the row already serves."""
        row = object.__getattribute__(self, "_row")
        xy = row.get("xy")
        if k in ("x", "y") and isinstance(xy, (list, tuple)) \
                and len(xy) == 2:
            return xy[0 if k == "x" else 1]
        return None

    def _column(self, k):
        """What ``r.k``, ``r[k]`` and ``r.get(k)`` read: the served column
        the spelling names, else r.x / r.y; ``_MISSING`` when neither."""
        rk = self._resolve_key(k)
        if rk is not None:
            return object.__getattribute__(self, "_row")[rk]
        d = self._derived(k)
        return _MISSING if d is None else d

    def _miss(self, k) -> str:
        """Why this key is not here and what writes it: a relation column
        exists only where a compare named its ref
        (``geom_program._AUTOATTACH_PREFIXES``), so the remedy is the relate.
        """
        row = object.__getattribute__(self, "_row")
        pre = next((p for p in _AUTOATTACH_PREFIXES
                    if isinstance(k, str) and k.startswith(p)), None)
        generic = (f"row {row.get('fid')} carries no {k!r}; its columns "
                   f"are {sorted(self.keys())[:12]}")
        if pre is None or pre == "color_":
            # a colour column is a PALETTE word, not a relate; the
            # off-palette word has its own loud error at the op layer
            return generic
        place = k[len(pre):]
        if pre in _STANDING_SIDE_PREFIXES:
            # a side of the VIEWER, so the remedy is a viewpoint
            return (f"row {row.get('fid')} carries no {k!r}: "
                    f"'{pre.strip()}' is a side of a VIEWPOINT, and none "
                    f"is standing here. stand(at=<place>, "
                    f"facing=<place-or-compass-word>) writes it.")
        return (
            f"row {row.get('fid')} carries no {k!r}: these rows were "
            f"never related to {place!r}. RELATE them and the column rides "
            f"the result -- rows.order(nearest={place!r}), or "
            f"find(<class>, near={place!r}). One relate writes "
            f"'distance to {place}' and 'facing {place}'; '<dir> of "
            f"{place}' rides a cardinal cut on the same place.")

    def __getattr__(self, k: str):
        v = self._column(k)
        if v is _MISSING:
            raise AttributeError(self._miss(k))
        return v

    def __getitem__(self, k):
        if isinstance(k, int) and k in (0, -1):
            return self                  # one row indexes as itself
        if isinstance(k, slice):
            return [self]
        v = self._column(k)
        if v is _MISSING:
            raise KeyError(self._miss(k))
        return v

    def get(self, k: str, default=None):
        v = self._column(k)
        if v is not _MISSING:
            return v
        # .get keeps its contract (a true miss returns the default), but a
        # miss that NEAR-matches a real column gets one receipt note per key
        # per block.
        mp = object.__getattribute__(self, "_mp")
        if mp is not None and isinstance(k, str):
            close = difflib.get_close_matches(
                k.replace("_", " "), self.keys(), n=1, cutoff=0.6)
            if close and k not in mp._get_noted:
                mp._get_noted.add(k)
                mp._fold({"note": (
                    f".get({k!r}) matched no column and returned the "
                    f"default; the near column is {close[0]!r}")})
        return default

    def keys(self) -> list:
        row = object.__getattribute__(self, "_row")
        return [k for k in row if k not in MODEL_UNSERVED_COLUMNS
                and not k.startswith("_")]

    # Value equality: a bare fid name mints a FRESH Row view per reference,
    # and two views of the same bind are the same place. A Row also equals
    # its own fid string.
    def _eq_key(self):
        b = self.bind
        return tuple(b) if isinstance(b, list) else b

    def __eq__(self, other):
        if isinstance(other, Row):
            return self._eq_key() == other._eq_key()
        if isinstance(other, str):
            return other == self.fid or other == self.bind
        return NotImplemented

    def __hash__(self):
        return hash(self._eq_key())

    def __contains__(self, k) -> bool:
        return k in self.keys()

    def __repr__(self):
        row = object.__getattribute__(self, "_row")
        cols = {k: row[k] for k in self.keys()
                if isinstance(row[k], (str, int, float, bool))
                or (isinstance(row[k], (list, tuple)) and len(row[k]) <= 4)}
        body = ", ".join(f"{k}={v!r}" for k, v in list(cols.items())[:10])
        return f"<row {body}>"


class RowList(list):
    """Rows the model narrowed in PYTHON: a plain list that still chains.

    A predicate keeps an arbitrary subset, which no executor ref spans. A
    list of :class:`Row` rides every slot element-wise, so the subset is
    first-class; the subclass only keeps ``.filter``/``.order`` chaining.
    """

    __slots__ = ()

    def _mp(self):
        """The mission the rows came from. A list carries no ref of its own,
        but every Row in it carries the surface it was minted on."""
        for r in self:
            mp = object.__getattribute__(r, "_mp")
            if mp is not None:
                return mp
        raise CodeError(
            "these rows are empty, so there is nothing to cut or order")

    def filter(self, pred=None, **kw) -> "RowList | Rows":
        # A PREDICATE FILTER IS NOT A DEAD END: the served cut works on a
        # materialised subset exactly as it does on a handle.
        _refuse_filter_kwargs(kw)
        _refuse_pred_and_cut(pred, kw)
        if pred is None:
            if not kw:
                raise CodeError(
                    "filter needs a predicate or near=/within=/dwithin=: "
                    "rows.filter(lambda r: r.distance(<place>) <= 25), or "
                    "rows.filter(near=<place>, dwithin=25).")
            return self._mp()._rows_filter(list(self), **kw)
        return RowList(r for r in self if pred(r))

    def order(self, key=None, **kw) -> "RowList | Rows":
        if key is None:
            # A GLOBAL ORDERING NEEDS NO SERVER, so it answers here rather
            # than compiling: the rows are in hand and the axis is the map's.
            if "direction" in kw:
                _refuse_reversed_direction(kw["direction"],
                                           kw.pop("reverse", False))
                if len(kw) > 1:
                    raise CodeError(
                        "order takes direction= OR nearest=/color=, not "
                        'both: direction="north" IS the key.')
                return RowList(sorted(self, key=_direction_key(
                    kw["direction"])))
            if not kw:
                raise CodeError(
                    "order needs a key or nearest=/color=/direction=: "
                    "rows.order(lambda r: r.distance(<place>)), "
                    'rows.order(nearest=<place>), or rows.order('
                    'direction="north") for the map-fixed order.')
            return self._mp()._rows_order(list(self), **kw)
        rev = bool(kw.pop("reverse", False))
        if kw:
            raise CodeError("order takes a key OR nearest=/color=/direction=, "
                            "not both")
        return RowList(sorted(self, key=key, reverse=rev))


class Rows:
    """A result set behind an executor ref. Lives for the whole mission.

    Indexing gives a :class:`Row`; a slice is a window over the same ref
    (order preserved); ``len()`` counts the rows the ref holds now.
    """

    __slots__ = ("ref", "spec", "mp", "label")

    def __init__(self, ref: str, mp: "MissionPython", spec: str = "",
                 label: str = ""):
        self.ref = ref
        self.spec = spec
        self.mp = mp
        self.label = label          # the call that made it, for refusals

    def _rows(self) -> list[dict]:
        rows = self.mp.env.rows_of_ref(self.ref)
        if self.spec:
            lo, hi = self.spec[1:-1].split(":")
            rows = rows[int(lo):int(hi)]
        return rows

    def __len__(self):
        return len(self._rows())

    def _tied(self) -> bool:
        return bool(not self.spec and self.mp.env.tied_of_ref(self.ref))

    def __getitem__(self, i):
        # A TIED VIEW IS NOT SUBSCRIPTABLE, the way ``dict.keys()`` is not:
        # every row ties, so an index would read as a choice and be a coin
        # flip.
        if self._tied():
            name = (self.mp.name_of(self) or self.mp.call_arg(None)
                    or self.ref)
            # The refusal names the call that does the work.
            raise CodeError(
                f"{name} is a tied view of {len(self._rows())} rows and is "
                "not subscriptable: they all match equally, so no row is "
                f"its first and no span is its middle. explore({name}) "
                "covers the whole set.")
        if isinstance(i, slice):
            lo = int(i.start or 0)
            if i.stop is None or i.step not in (None, 1) or lo < 0:
                raise CodeError("a window is [lo:hi] with both ends")
            hi = int(i.stop)
            if self.spec:
                # a slice of a slice composes into absolute indices
                base = int(self.spec[1:-1].split(":")[0])
                lo, hi = base + lo, base + hi
            return Rows(self.ref, self.mp, f"[{lo}:{hi}]")
        rows = self._rows()
        if not isinstance(i, int) or not (-len(rows) <= i < len(rows)):
            name = (self.mp.name_of(self) or self.mp.call_arg(None)
                    or self.ref)
            raise CodeError(
                f"{name}[{i}] did not bind: {name} served {len(rows)} "
                "row(s) here. Show it first, then index it.")
        return Row(rows[i], src=(self.ref, i % len(rows)), mp=self.mp)

    def __iter__(self):
        return (Row(r, src=(self.ref, i), mp=self.mp)
                for i, r in enumerate(self._rows()))

    # THE MATCH TIERS AS ATTRIBUTES: the agent chooses a find's stratum by
    # holding it -- r1.exact, explore(r1.exact + r1.tagged). Live views; the
    # full set stays behind this handle.
    @property
    def exact(self) -> "Rows":
        """The rows that ARE the queried thing -- the top match tier."""
        return self._stratum("exact")

    @property
    def tagged(self) -> "Rows":
        """The tag-carriers -- rows matched through a key=value tag."""
        return self._stratum("tagged")

    @property
    def fuzzy(self) -> "Rows":
        """The word-sharers -- the fuzz band below the exact tiers."""
        return self._stratum("fuzzy")

    def _stratum(self, tier: str) -> "Rows":
        src = self if not self.spec else self.mp._materialize(self)
        p = self.mp._program(src.ref)
        return Rows(self.mp._analyze(
            p, p.add("stratum", {"items": src.ref, "tier": tier}),
            shown=f"{src.ref}.{tier}"), self.mp)

    # -- cut and order: many filters, ONE ordering ------------------------
    #
    # The kwarg form is SERVED: it compiles to a `distance to <anchor>`
    # column and a cut on it, a footprint gap to a set's nearest member.
    # The predicate form runs any Python over the rows in hand.
    def filter(self, pred=None, *, near=None, within=None, dwithin=None,
               north_of=None, south_of=None, east_of=None, west_of=None,
               facing=None, direction=None,
               **kw) -> "Rows | RowList":
        """Keep the rows that hold. Filters compose.

        Every cut is a relation to a place, so each takes its anchor (one
        place or a set): ``east_of=`` keeps what lies that way of it,
        ``facing=`` with ``direction=`` keeps what FACES it that way.
        """
        _refuse_filter_kwargs(kw)
        _refuse_metres_in_within(within)
        _refuse_pred_and_cut(pred, dict(
            near=near, within=within, dwithin=dwithin, facing=facing,
            direction=direction, north_of=north_of, south_of=south_of,
            east_of=east_of, west_of=west_of))
        if pred is not None:
            if not callable(pred):
                raise CodeError(
                    f"filter's first argument is a predicate and {pred!r} "
                    "is not callable: rows.filter(lambda r: r.distance(x) <= 25)")
            return RowList(r for r in self if pred(r))
        src = self if not self.spec else self.mp._materialize(self)
        return self.mp._rows_filter(
            src, within=within, near=near,
            dwithin=dwithin, north_of=north_of, south_of=south_of,
            east_of=east_of, west_of=west_of,
            facing=facing, direction=direction)

    def order(self, key=None, *, nearest=None, color=None,
              direction=None, reverse=False) -> "Rows | RowList":
        """Put the rows in one order; a table sorts by one column.
        ``direction=`` and ``key=`` sort the rows in hand (``reverse=`` flips a
        key sort, as ``sorted()`` does); ``nearest=`` and ``color=`` re-rank
        the whole set server-side, and relating to a place stamps ``facing
        <place>`` and ``distance to <place>``."""
        if key is not None or direction is not None:
            if key is not None and direction is not None:
                raise CodeError(
                    'order takes a key OR direction=, not both: '
                    'direction="north" IS the key.')
            if nearest is not None or color is not None:
                raise CodeError("order takes a key OR nearest=/color=, "
                                "not both")
            if direction is not None:
                _refuse_reversed_direction(direction, reverse)
                return RowList(sorted(self, key=_direction_key(direction)))
            if not callable(key):
                raise CodeError(
                    f"order's first argument is a key function and {key!r} "
                    "is not callable: rows.order(lambda r: r.distance(x))")
            return RowList(sorted(self, key=key, reverse=bool(reverse)))
        src = self if not self.spec else self.mp._materialize(self)
        return self.mp._rows_order(src, nearest=nearest, color=color)

    def distance(self, other) -> float:
        """Metres to ``other``, this set counting as ONE place: the free
        ``distance(a, b)`` as a method."""
        return self.mp._call_distance(self, other)

    def __add__(self, other):
        # Set union, the Python way: a list of rows to loop and read.
        if isinstance(other, Rows):
            return RowList(list(self) + list(other))
        if isinstance(other, list):
            return RowList(list(self) + other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, list):
            return RowList(other + list(self))
        return NotImplemented

    def __repr__(self):
        """Named rows, not an opaque count."""
        name = self.mp.name_of(self) or self.ref
        rows = self._rows()
        if not rows:
            return f"<{name}: 0 rows>"

        def _one(r):
            fid = r.get("fid")
            xy = r.get("xy")
            if isinstance(xy, (list, tuple)) and len(xy) == 2:
                return f"{fid} [{xy[0]:g}, {xy[1]:g}]"
            return str(fid)

        if self._tied():
            # A TIED VIEW SHOWS ITS SHAPE, as the served head does: no
            # fids and no points.
            _nm = str((rows[0] or {}).get("name") or name)
            return f"<{_nm}: {len(rows)} ways>"
        head = ", ".join(_one(r) for r in rows[:3])
        more = f", +{len(rows) - 3} more" if len(rows) > 3 else ""
        return f"<{name}: {len(rows)} row(s): {head}{more}>"


# --------------------------------------------------------------------------
# The measuring calls: footprint geometry between places
# --------------------------------------------------------------------------


# One block's measuring budget: a nested loop of distance() calls is one per
# PAIR, millions of mask-to-mask scans at real set sizes.
MEASURE_CALL_CEILING = 20_000


def _tuple_xy(xy):
    """A served ``xy`` as a hashable pair, or None. The place tuple is the
    geometry cache's key, so a list would make it unhashable."""
    return (float(xy[0]), float(xy[1])) if is_xy(xy) else None


def _set_as_list(v) -> list:
    """A Python SET of things is the list of them, in a fixed order."""
    return sorted(v, key=lambda x: str(x.bind) if isinstance(x, Row)
                  else repr(x))


def _places(v, who: str) -> tuple:
    """A Python value -> the ``(fid, xy)`` places geometry measures over. A set
    is one place meaning its nearest member; the xy rides for a computed row (a
    crossing, a terminus), whose point is its whole footprint."""
    if isinstance(v, Row):
        return ((v.fid, _tuple_xy(v.get("xy"))),)
    if isinstance(v, Rows):
        return tuple((r.get("fid"), _tuple_xy(r.get("xy")))
                     for r in v._rows())
    if is_xy(v):
        return ((None, (float(v[0]), float(v[1]))),)
    if isinstance(v, (list, tuple, set, frozenset)):
        if not v:
            raise CodeError(f"{who}() got an empty list -- there is "
                            "nothing there to measure.")
        return tuple(p for x in v for p in _places(x, who))
    if isinstance(v, str):
        if _fid_like(v):
            return ((v, None),)
        # A NAME HAS NO GEOMETRY until find retrieves it.
        raise CodeError(
            f"{who}() measures between things on the surface, and {v!r} is "
            f"a name. find({v!r}) first, then measure against the rows it "
            "serves -- a set is one place and means its nearest member.")
    raise CodeError(
        f"{who}() takes a row, a result set, a fid or an [x, y] point; got "
        f"{type(v).__name__} ({v!r}).")


# --------------------------------------------------------------------------
# The mission: one namespace, the library, the block runner
# --------------------------------------------------------------------------


class _Name(str):
    """A place spelled by NAME: grounded by the loop before the call runs."""


def _tok(site, ties: dict) -> str:
    """A site's token once its names are grounded: the best name match."""
    return ties[site][0] if isinstance(site, _Name) else site


def _plain(v):
    """An argument as the model spelled it: a result or a row is its label."""
    if isinstance(v, (Ref, Bind)):
        return v.label
    if isinstance(v, (list, tuple)):
        return type(v)(_plain(x) for x in v)
    return v


def _look_tok(a) -> str:
    """One element of ``look([a, b, c])`` -> the token its crop centres on."""
    if isinstance(a, Bind):
        b = a.bind
        if isinstance(b, (list, tuple)) and len(b) == 2:
            return format_xy_token(float(b[0]), float(b[1]))
        return b if isinstance(b, str) else a.label
    if isinstance(a, Ref):
        return a.label
    if is_xy(a):
        return format_xy_token(float(a[0]), float(a[1]))
    return str(_plain(a))


@dataclass
class MissionEnv:
    """The loop's verbs, one set per mission. A library call opens with
    ``ground(names)`` (``{name: why}`` -> ``{"ties": {name: fids}}`` or an
    error), runs one verb and, unless refused, closes with ``finish(ack)``; a
    refusing ack part carries ``"error"``. ``plan_rn`` is the shared ref
    counter, ``widened`` the paging ledger, ``minted`` the handles the call's
    flights minted, ``say_err(ack, msg)`` the refusal format,
    ``row_of_fid(fid)`` a live-row reader (None when absent) and
    ``tied_of_ref(ref)`` whether a ref holds a tied view."""

    rows_of_ref: Callable
    plan_rn: list
    widened: dict
    minted: list
    say_err: Callable
    ground: Callable
    analyze: Callable
    stand: Callable
    move: Callable
    cover: Callable
    crops: Callable
    map: Callable
    watch: Callable
    boost: Callable
    finish: Callable
    row_of_fid: Callable
    tied_of_ref: Callable = lambda _ref: False


class MissionPython:
    """The persistent interpreter for one mission's run channel."""

    def __init__(self, env: MissionEnv):
        self.env = env
        self.ns: dict = MissionNamespace(
            {"__builtins__": dict(_SAFE_BUILTINS)})
        self.ns.mp = self
        self.ns["__builtins__"]["print"] = self._print
        self.ns["__builtins__"]["__import__"] = _safe_import
        self.ns["__show__"] = self._show
        # show(x) is the bare-expression renderer under the model's own
        # display verb.
        self.ns["show"] = self._call_show
        self.ns.update(_SAFE_MODULES)
        for name in served_calls():
            self.ns[name] = self._wrap_call(name)
        # Model spelling for every ref it was shown rows under, so the
        # served envelope is keyed by the model's own name.
        self._shown: dict = {}
        self._labels: dict = {}         # ref -> the call that made it (find)
        self._picked: dict = {}         # fids the model's Row names hold
        self._unbound: dict = {}        # name -> why this mission removed it
        self._get_noted: set = set()    # .get near-miss keys noted, per block
        self._call_minted: list = []    # handles the last call's flights minted
        # Footprint geometry BY FID off the live surface. The cache is
        # per-block: a flight folds detections in under it, so last block's
        # union would measure a world that no longer stands.
        self._geom = SurfaceGeometry(env.row_of_fid)
        self._measure_calls = 0
        # per-block state
        self._out: dict = {}
        self._code = ""
        self._printed: list = []
        self._lines = 0
        self._deadline = 0.0
        self._block_shown: set = set()

    def _wrap_call(self, name: str):
        """The library callable the namespace serves: the real
        ``_call_*``, with signature TypeErrors re-said without the internal
        name and with the keywords that DO exist."""
        fn = getattr(self, f"_call_{name}")
        kws = ", ".join(
            f"{p.name}=" for p in inspect.signature(fn).parameters.values()
            if p.kind is p.KEYWORD_ONLY
            or p.default is not p.empty) or "none"

        def call(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except TypeError as e:
                if "_call_" not in str(e):
                    raise               # a real TypeError from inside
                msg = re.sub(r"\S*_call_(\w+)\(\)", r"\1()", str(e))
                raise CodeError(
                    f"{msg}. {name} keywords: {kws}") from None

        call.__name__ = name
        return call

    # -- naming ------------------------------------------------------------

    def name_of(self, obj) -> str | None:
        for k, v in self.ns.items():
            if v is obj and not k.startswith("__"):
                return k
        return getattr(obj, "label", "") or None

    def ref_name(self, ref: str) -> str | None:
        """The model's name for a whole result: a bound name, else what it
        was shown under or the call that made it; None when it has none."""
        for k, v in self.ns.items():
            if isinstance(v, Rows) and not v.spec and v.ref == ref \
                    and not k.startswith("__"):
                return k
        return self._shown.get(ref) or self._labels.get(ref)

    def call_arg(self, fn: str | None) -> str | None:
        """The model's text for the first argument of the ``fn(...)`` call
        its block is running (``fn`` None: for what the subscript it is
        running indexes), found at the block frame's live position."""
        fr = sys._getframe()
        while fr is not None and fr.f_code.co_filename != BLOCK_FILENAME:
            fr = fr.f_back
        if fr is None:
            return None
        at = inspect.getframeinfo(fr, context=0).positions
        for node in ast.walk(ast.parse(self._code)):
            if not isinstance(node, (ast.Call, ast.Subscript)) or (
                    node.lineno, node.col_offset) != (at.lineno, at.col_offset):
                continue
            if fn is None and isinstance(node, ast.Subscript):
                return ast.unparse(node.value)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == fn and node.args):
                return ast.unparse(node.args[0])
        return None

    def _held(self, name: str):
        """What a bound name holds, as an argument: a whole result, a row,
        or a staged viewpoint; None for anything else."""
        v = dict.get(self.ns, name)
        if isinstance(v, Rows) and not v.spec:
            return Ref(v.ref, name)
        if isinstance(v, Row):
            return Bind(v.bind, name)
        if isinstance(v, str) and v == name and re.fullmatch(r"v\d{1,3}", v):
            return Bind(v, name)
        return None

    def _slot(self, v):
        """A Python value -> the argument the compiler reads: a result is a
        :class:`Ref` and a row a :class:`Bind`, labelled with the model's
        own name (as is a string naming one); anything else rides verbatim,
        a bare name to ground."""
        if isinstance(v, Rows):
            if v.spec:
                v = self._materialize(v)
            return Ref(v.ref, self.name_of(v) or v.ref)
        if isinstance(v, Row):
            return Bind(v.bind, self.name_of(v) or (v.fid or "row"))
        if isinstance(v, str):
            return self._held(v) or v
        if isinstance(v, (set, frozenset)):
            v = _set_as_list(v)
        if isinstance(v, (list, tuple)) and not is_xy(v):
            # a LIST of things -- order(nearest=[roadA, roadB]) -- maps
            # element-wise.
            out = [self._slot(x) for x in v]
            return out if any(isinstance(x, (Row, Rows)) for x in v) \
                else type(v)(out)
        return v

    def _program(self, *args) -> Program:
        """A program for one call: it mints past the rN names the call spells
        and the mission's handles hold."""
        taken = rn_spelled(args) | {
            k for k, v in self.ns.items() if re.fullmatch(r"r\d+", k)
            and (isinstance(v, Row) or isinstance(v, Rows) and not v.spec)}
        live = {v.ref for v in self.ns.values()
                if isinstance(v, Rows) and not v.spec}
        return Program(self.env.plan_rn[0], taken, live)

    # -- the per-call plumbing --------------------------------------------

    def _end(self, ack: dict) -> None:
        """Close one call: finish an ack nothing refused, fold it into the
        receipt, bind what its flights minted; a refusal ends the block."""
        if "error" not in ack:
            ack = self.env.finish(ack)
        self._fold(ack)
        self._call_minted = list(self.env.minted)
        for h in self._call_minted:
            self.ns[h] = Rows(h, self)
        del self.env.minted[:]
        if "error" in ack:
            raise _Stop()

    def _ground(self, sites) -> dict:
        """Open a call: ground the NAMES among its ``(site, why)`` pairs ->
        ``{name: top-tie fids}``."""
        why: dict = {}
        for n, w in sites:
            if isinstance(n, _Name):
                why.setdefault(n, w)
        got = self.env.ground(why)
        if "error" in got:
            self._end(got)
        return got["ties"]

    def _site(self, v, why: str):
        """An act's PLACE -> its token, or a :class:`_Name` to ground. A
        point rides as its ``[x, y]`` token; what a binding holds and a fid
        the model typed are never looked up."""
        if isinstance(v, Ref):
            return v.ref
        if isinstance(v, Bind):
            b = v.bind
            if isinstance(b, (list, tuple)) and len(b) == 2:
                return format_xy_token(float(b[0]), float(b[1]))
            return b if isinstance(b, str) else str(v.label)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            v = _plain(v)
            return format_xy_token(float(v[0]), float(v[1]))
        # A QUOTED [x, y] IS A POINT: welcome in a relating slot, refused as
        # a COORDINATE in a moving one.
        pt = parse_xy_token(v) if isinstance(v, str) else None
        if pt:
            if why in ("move", "cover") and not withheld("prior"):
                raise CodeError(act_needs_a_thing(
                    name="go_to/explore", got=f"{pt[0]:g}, {pt[1]:g}")
                    + " A crossing you computed is a ROW: index it "
                      "-- r2 = intersection(\"A Road\", \"B St\") then "
                      "go_to(r2[0]).")
            return format_xy_token(float(pt[0]), float(pt[1]))
        if isinstance(v, str) and not _fid_like(v):
            return _Name(v)
        return str(_plain(v))

    def _cards(self, **anchors) -> list:
        """The cardinal cuts given, slotted in the order named, read in
        the compiler's order."""
        got = {d: self._slot(a) for d, a in anchors.items() if a is not None}
        return [(d, got[d]) for d in _CARDINALS if d in got]

    def _act(self, verb, sites: list, *args, **kw) -> None:
        """Ground the NAMES among ``sites`` (``(site, why)`` pairs), then run
        the act ``verb`` on their tokens."""
        ties = self._ground(sites)
        self._end(verb(*[_tok(s, ties) for s, _w in sites], *args, **kw))

    def _show_as(self, ref: str, label: str) -> None:
        """Serve ``ref``'s envelope under ``label``, kept whole."""
        self._shown[ref] = label
        self._block_shown.add(ref)

    def _analyze(self, p: Program, ref: str, *, shown=None, page=None) -> str:
        """Ground the program's named anchors and run it, ``ref``'s envelope
        the answer (served under ``shown`` when given)."""
        if shown is not None:
            self._show_as(ref, shown)
        ties = self._ground((_Name(n), "anchor") for n in p.anchor_names())
        p.ground(ties)
        self._end(self.env.analyze(p.ops, p.rn, ref, page))
        return ref

    def _fold(self, part: dict) -> None:
        out = self._out
        for k, v in part.items():
            if k == "said":
                continue          # the receipt's said is the whole block
            if k == "result" and isinstance(out.get("result"), dict) \
                    and isinstance(v, dict):
                out["result"] = {
                    **out["result"], **v,
                    "results": {**(out["result"].get("results") or {}),
                                **(v.get("results") or {})}}
            elif k == "note" and out.get("note"):
                out["note"] = f"{out['note']}\n{v}"
            else:
                out[k] = v

    def _window(self, ref: str, spec: str, label: str, *args) -> str:
        """The identity window over ``ref`` (sliced by ``spec``), served
        under ``label``."""
        p = self._program(*args)
        return self._analyze(p, p.window(ref, spec), shown=label)

    def _materialize(self, sliced: "Rows") -> "Rows":
        """A sliced set becomes a real ref via the identity window, so a
        slice rides every slot a set can (find(near=r1[0:5]) is the union
        of the head, which is what the spelling plainly means)."""
        return Rows(self._window(sliced.ref, sliced.spec,
                                 f"{sliced.ref}{sliced.spec}", sliced.ref),
                    self)

    # -- the library -------------------------------------------------------

    def _call_find(self, query="", *more, nearest=None, near=None,
                   within=None, dwithin=None, color=None,
                   name=None, north_of=None, south_of=None, east_of=None,
                   west_of=None):
        if more:
            raise CodeError(
                'find takes ONE query -- find("89th Street") then '
                'find("Fuchsia Avenue"), or intersection("89th Street", '
                '"Fuchsia Avenue") for where they cross.')
        # near=[one thing] is the thing
        if isinstance(nearest, (list, tuple)) and len(nearest) == 1 \
                and not is_xy(nearest):
            nearest = nearest[0]
        if isinstance(near, (list, tuple)) and len(near) == 1 \
                and not is_xy(near):
            near = near[0]
        _refuse_metres_in_within(within)
        if isinstance(nearest, (list, tuple)) and not is_xy(nearest):
            nearest = [self._slot(a) for a in nearest]
        elif nearest is not None:
            nearest = self._slot(nearest)
        near = self._slot(near) if near is not None else None
        within = self._slot(within) if within is not None else None
        cards = self._cards(north=north_of, south=south_of, east=east_of,
                            west=west_of)
        q = str(query) if query != "" else ""
        p = self._program(q, nearest, near, within, dwithin, color, name,
                          cards)
        ref = compile_find(p, q, name=name, near=near, nearest=nearest,
                           dwithin=dwithin, within=within, cards=cards,
                           color=color)
        self._analyze(p, ref)
        self._near_miss_note(ref, query)
        if q:
            self._labels[ref] = f"find({query!r})"
        return Rows(ref, self, label=f"find({query!r})" if q else "")

    # find("pool", nearest=<street>) re-sorts a global name-match, so pools
    # 2.5 km away can read like "the pools near the street"; this note says
    # none of what was found is NEAR the anchor.
    _NEAR_MISS_M = 500.0

    def _near_miss_note(self, ref, query) -> None:
        rows = self.env.rows_of_ref(ref) if ref else []
        if not rows:
            return
        dkey = next((k for k in rows[0]
                     if isinstance(k, str) and k.startswith("distance to ")),
                    None)
        if dkey is None:
            return
        vals = [r.get(dkey) for r in rows
                if isinstance(r.get(dkey), (int, float))]
        if not vals or min(vals) <= self._NEAR_MISS_M:
            return
        anchor = dkey[len("distance to "):]
        self._fold({"note": (
            f"nothing found for {str(query)!r} lies near {anchor} -- the "
            f"nearest match is {min(vals):,.0f} m away."
            + ("" if withheld("look_for") else
               f" Rows of a watched class appear near {anchor} only when a "
               "flight covers it."))})

    def _rows_order(self, items, *, nearest=None, color=None):
        """One sort key, never two: any way of combining two keys can rank a
        row that is second on both below worse rows, so a second key is refused
        and named as a cut."""
        if nearest is not None and color is not None:
            raise CodeError(
                "order takes ONE key: a table sorts by one column. "
                "nearest= and color= are two, and combining them ranks "
                "what is mediocre at both above what is right. Cut on one "
                "-- rows.filter(near=<place>, dwithin=<metres>) -- then "
                "order the survivors by the other.")
        if nearest is None and color is None:
            raise CodeError(
                "order needs its key: rows.order(nearest=<place>) by "
                "distance, rows.order(color=\"red\") by colour, or "
                "rows.order(<function>) for one you compute yourself.")
        if isinstance(nearest, (list, tuple)) and not is_xy(nearest):
            # nearest=[A, B] is a SET -- one place, its nearest member --
            # not two keys to average.
            nearest = [self._slot(a) for a in nearest]
        elif nearest is not None:
            nearest = self._slot(nearest)
        items = self._slot(items)
        p = self._program(items, nearest, color)
        return Rows(self._analyze(p, compile_order(
            p, items, nearest=nearest, color=color)), self)

    def _rows_filter(self, items, *, within=None, near=None,
                     dwithin=None, facing=None, direction=None, **cards):
        # Which-arg validation lives in the compiler; every caller has
        # already refused a kwarg outside the set.
        _refuse_metres_in_within(within)
        items = self._slot(items)
        within = self._slot(within) if within is not None else None
        near = self._slot(near) if near is not None else None
        # facing= is a PLACE and rides the slot resolver; direction= is the
        # compass WORD and rides verbatim, which is the whole point of
        # keeping the word out of the kwarg name.
        facing = self._slot(facing) if facing is not None else None
        cards = self._cards(**{d[:-3]: v for d, v in cards.items()})
        p = self._program(items, within, near, dwithin, facing, direction,
                          cards)
        return Rows(self._analyze(p, compile_filter(
            p, items, within=within, near=near, dwithin=dwithin,
            facing=facing, direction=direction, cards=cards)), self)

    def _call_intersection(self, a, b):
        a, b = self._slot(a), self._slot(b)
        p = self._program(a, b)
        return Rows(self._analyze(p, p.add("crossings", {
            "refs": [p.ref_of(a), p.ref_of(b)]})), self)

    def _call_ends_of(self, x):
        x = self._slot(x)
        p = self._program(x)
        return Rows(self._analyze(p, p.add("ends", {
            "refs": [p.ref_of(x)]})), self)

    def _call_more_of(self, x):
        if not isinstance(x, Rows) or x.spec:
            raise CodeError("more_of pages a RESULT a find or an ordering "
                            "left")
        name = self.name_of(x) or self.call_arg("more_of") or x.ref
        lo = _BROAD_HEAD_ROWS + int(self.env.widened.get(x.ref) or 0)
        p = self._program(name)
        # The rows served are NOT the handle -- they are a window of it --
        # so the envelope is labelled with what was asked for rather than
        # with a name that would claim to be the whole set.
        return Rows(self._analyze(
            p, p.window(x.ref, f"[{lo}:{lo + _BROAD_HEAD_ROWS}]"),
            shown=f"more_of({name})", page=(x.ref, lo, name)), self)

    # -- measuring ---------------------------------------------------------

    def _call_distance(self, a, b) -> float:
        """Metres between two footprints, mask to mask; 0.0 on touch or
        containment. Either side may be a set, meaning its nearest member."""
        self._measure_calls += 1
        if self._measure_calls > MEASURE_CALL_CEILING:
            raise CodeError(
                f"this block ran past {MEASURE_CALL_CEILING:,} measurements. "
                "distance() in a loop inside a loop is one call per PAIR, "
                "which is millions over sets this size. Slice the sets "
                "down, or measure each thing against the WHOLE other set "
                "in one call -- a set is one place and means its nearest "
                "member.")
        return self._geom.distance(_places(a, "distance"),
                                   _places(b, "distance"))

    def _call_look(self, x, *, marks=None):
        # marks= names the rows whose identity bears on the decision; a
        # named road is drawn as its NAME, everything else outlines and
        # badges.
        mk = None
        if marks is not None:
            mk = [self._slot(m) for m in (
                marks if isinstance(marks, (list, tuple)) else [marks])]
        # The marks ride the SAME crop request as its subject, so the
        # renderer's frame decision and its mark projection see one window.
        def msites():
            return [(self._site(m, "marks"), "marks") for m in mk or []]

        def crops(ties, marks, of=None, fid=None):
            return self.env.crops(of, fid, [_tok(s, ties) for s, _w in marks]
                                  if mk else None)

        if isinstance(x, Rows) and x.spec:
            # A SLICE is looked at through its window: two verbs, one
            # receipt, and the window is no answer of its own.
            name = self.name_of(x) or self.ref_name(x.ref) or x.ref
            p = self._program(name, mk)
            ref = p.window(x.ref, x.spec)
            self._show_as(ref, f"{name}{x.spec}")
            marks = msites()
            ties = self._ground(marks)
            ack = self.env.analyze(p.ops, p.rn, None, None)
            if "error" not in ack:
                ack.update(crops(ties, marks, of=ref))
            self._end(ack)
            return None
        x = self._slot(x)
        if isinstance(x, (list, tuple)) and not is_xy(x):
            # look([a, b, c]) crops EACH element.
            fids = [_look_tok(a) for a in x]
            marks = msites()
            self._end(crops(self._ground(marks), marks, fid=fids))
        elif isinstance(x, Ref):
            # A SET look is the badged frame over its rows.
            marks = msites()
            self._end(crops(self._ground(marks), marks, of=x.ref))
        else:
            site = self._site(x, "crops")
            marks = msites()
            ties = self._ground([(site, "crops")] + marks)
            self._end(crops(ties, marks, fid=_tok(site, ties)))
        return None

    def _call_show_map(self, around=None):
        if isinstance(around, Rows):
            name = self.name_of(around) or around.ref
            raise CodeError(
                f"around= takes ONE place and {name} is a set. Index it "
                f"-- around={name}[0] -- or use a name. (A set IS one "
                "place in near=, within= and nearest=, where it means "
                "the gap to its NEAREST member, but there is no one "
                "point to centre on.)")
        around = self._slot(around) if around is not None else None
        self._act(self.env.map, [(self._site(around, "map"), "map")])
        return None

    def _call_look_for(self, *classes, at=None):
        # at=, not near=: this names WHERE to segment extra.
        if not classes:
            raise CodeError('look_for takes the classes to watch: '
                            'look_for("house", "pool")')
        if not all(isinstance(c, str) for c in classes):
            raise CodeError('look_for takes class names as strings: '
                            'look_for("house", "pool")')
        # THE SEGMENTER CHANNEL: it arms perception and moves nothing. The
        # segmenter reads WORDS, so look_for("parking_lot") and
        # look_for("parking lot") arm the same class.
        words = [w.strip().lower().replace("_", " ")
                 for w in classes if w.strip()]
        at = self._slot(at) if at is not None else None
        if at is None:
            self._act(self.env.watch, [], words)
        else:
            # The placed form is the TARGETED BOOST, and the loop arms the
            # standing watchlist from it too -- one verb, one semantics.
            self._act(self.env.boost, [(self._site(at, "boost"), "boost")],
                      words)
        return None

    def _call_stand(self, *, at, facing="north"):
        if isinstance(at, Rows):
            raise CodeError("at= takes ONE place and a set has no one "
                            "point to stand at. Index it -- at=rows[0].")
        at = self._slot(at)
        if not isinstance(facing, str):
            facing = self._slot(facing)
        # A heading: a compass word rides as itself, anything else is a
        # place to look TOWARD.
        at_site = self._site(at, "stage")
        if isinstance(facing, str) and facing.strip().lower() in COMPASS_UNIT:
            f_site = facing.strip().lower()
        else:
            f_site = self._site(self._held(facing) or facing
                                if isinstance(facing, str) else facing,
                                "facing")
        self._act(self.env.stand, [(at_site, "stage"), (f_site, "facing")])
        staged = (self._out.get("stance") or {}).get("staged") or []
        for s in staged:
            h = s.get("handle")
            if isinstance(h, str):
                self.ns[h] = h
        return staged[-1].get("handle") if staged else None

    def _move(self, call: str, x):
        if isinstance(x, Rows):
            # A SET is one place for explore; go_to lands on one thing.
            if call == "go_to":
                name = self.name_of(x) or x.ref
                raise CodeError(
                    f"go_to() takes ONE thing and {name} is a set of "
                    f"{len(x)} row(s). Index it -- go_to({name}[0]) -- "
                    f"or explore({name}) to cover the whole set.")
            if not len(x):
                name = self.name_of(x) or x.ref
                raise CodeError(
                    f"{name} holds 0 row(s) here; there is nothing to "
                    "cover.")
        elif is_xy(x) and withheld("prior"):
            pass  # no map to name a thing on: a point is the place to fly
        elif isinstance(x, (list, tuple, set, frozenset)):
            if isinstance(x, (set, frozenset)):
                x = _set_as_list(x)
            # explore's list is a SET of things -- rows, sets, fids -- as
            # one place to cover; any numeric element is the coordinate
            # hatch.
            things = (bool(x) and not is_xy(x)
                      and all(isinstance(e, (Row, Rows, str)) for e in x))
            ok_set = call == "explore" and things
            # A SET OF THINGS IS NOT A COORDINATE: go_to refuses it as a set.
            if things and not ok_set:
                raise CodeError(
                    f"{call}() takes ONE thing and that is a set of "
                    f"{len(x)}. Index the one you mean -- {call}(<set>[0]) "
                    "-- or explore(<set>) to cover all of it.")
            if not ok_set:
                raise CodeError(act_needs_a_thing(
                    name="go_to/explore",
                    got=", ".join(f"{v:g}" if isinstance(v, (int, float))
                                  else str(v) for v in x)))
        if isinstance(x, str) and re.fullmatch(r"v\d{1,3}", x):
            # A staged viewpoint is a pose, not a thing on the map: the
            # move wire would resolve it to its stand point, a coordinate.
            raise CodeError(
                f"{x} is a viewpoint you adopted, not a thing on the "
                f"map, and {call}() takes a thing. A stance is for "
                "looking -- its sides ride every find row. To move, "
                f"{call}() a row: r1[0], or a fid off one.")
        x = self._slot(x)
        if call == "go_to":
            self._act(self.env.move, [(self._site(x, "move"), "move")])
        else:
            # A `cover` move flies a pass over the thing and ends on its
            # last stop. A SET is one place to cover; a NAME covers its
            # whole top tie group.
            many = isinstance(x, (list, tuple)) and not is_xy(x)
            sites = [self._site(a, "cover") for a in (x if many else [x])]
            said = _plain(x)
            ties = self._ground((s, "cover") for s in sites)
            self._end(self.env.cover(
                [ties[s] if isinstance(s, _Name) and len(ties[s]) > 1
                 else _tok(s, ties) for s in sites],
                said if isinstance(said, str)
                else ", ".join(str(a) for a in said)))
        # The arrivals: a flight minted a handle over the rows it added,
        # so `r = explore(x)` binds them.
        return self.ns.get(self._call_minted[-1]) if self._call_minted \
            else None

    def _call_go_to(self, x):
        return self._move("go_to", x)

    def _call_explore(self, x):
        return self._move("explore", x)

    # -- display -----------------------------------------------------------

    def _print(self, *parts, sep=" ", end=""):
        self._printed.append(sep.join(str(p)[:2_000] for p in parts))

    def _call_show(self, *xs):
        """``show(r1)`` renders exactly what ``r1`` on its own line does."""
        for x in xs:
            self._show(x, self.name_of(x)
                       or (x.ref if isinstance(x, Rows) else repr(x)))

    def _show(self, value, label: str):
        if value is None:
            return
        if isinstance(value, Rows):
            name = self.name_of(value) or value.ref
            # Key the envelope by the MODEL's spelling ("r1[0:5]"), not the
            # executor's ref the window compiled onto.
            self._shown[self._window(value.ref, value.spec,
                                     f"{name}{value.spec}", name)] = label
        elif isinstance(value, Row) and value.src \
                and not re.fullmatch(r"\w+", label):
            # Render the ROW: a one-row window keeps the ordering
            # attribute a bare dict loses, so the model sees the row it
            # asked for rather than a census of one.
            ref, i = value.src
            spec = f"[{i}:{i + 1}]"
            self._shown[self._window(ref, spec, f"{ref}{spec}", ref)] = label
        elif isinstance(value, Row):
            self._out.setdefault("shown", {})[label] = value.bind
        else:
            self._printed.append(f"{label} = {value!r}"
                                 if label != repr(value) else repr(value))

    # -- the block runner --------------------------------------------------

    def run_block(self, code: str) -> dict:
        self._out = out = {"said": code.strip()}
        self._code = code
        self._printed = []
        self._lines = 0
        self._block_shown: set = set()
        self._get_noted.clear()
        self._geom.clear()
        self._measure_calls = 0
        try:
            tree = _instrument(scan_block(code))
            compiled = compile(tree, BLOCK_FILENAME, "exec")
        except CodeError as e:
            return self.env.say_err(out, str(e))

        self._deadline = time.monotonic() + TRACE_DEADLINE_S
        try:
            with _block_ceilings(self, compiled):
                exec(compiled, self.ns)        # noqa: S102 -- the surface
        except _Stop:
            pass                     # the refusal is already in the receipt
        except CodeError as e:
            self._unbind_failed_assign(tree, _block_line())
            out.update(self.env.say_err({}, f"line {_block_line()}: {e}"))
        except Exception as e:                          # noqa: BLE001
            self._unbind_failed_assign(tree, _block_line())
            out.update(self.env.say_err(
                {}, f"line {_block_line()}: {type(e).__name__}: {e}"))
        return self._finish(out)

    def _tick(self) -> None:
        """One line of the block's OWN code ran: the liveness ceilings."""
        self._lines += 1
        if self._lines > TRACE_LINE_CEILING:
            raise CodeError(
                f"the block ran past {TRACE_LINE_CEILING:,} of its "
                "own lines -- an unbounded loop. Simplify the block.")
        if time.monotonic() > self._deadline:
            raise CodeError(
                f"the block ran past {TRACE_DEADLINE_S:.0f} s -- a slow or "
                "large query, most likely a find/near loop over the whole "
                "map. Scope it tighter (fewer rows, a smaller dwithin) or "
                "split it across turns.")

    def _unbind_failed_assign(self, tree: ast.Module, line: int) -> None:
        """A rebind that failed UNBINDS the name.

        Python leaves the target holding its OLD value when the right-hand
        side raises, and a stale pick that stays bound is a wrong commit.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and node.lineno == line:
                for t in node.targets:
                    # ast.walk reaches Names inside tuple/list unpacking
                    # too -- `a, b = <raises>` must unbind BOTH.
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            self.ns.pop(n.id, None)
                            self._unbound[n.id] = (
                                "its assignment failed, so the stale "
                                "value was removed -- re-derive it")

    def _finish(self, out: dict) -> dict:
        if self._printed:
            lines = self._printed
            if len(lines) > PRINT_LINE_CAP:
                kept = lines[:PRINT_LINE_CAP - 8] + [
                    f"... [{len(lines) - PRINT_LINE_CAP + 3} printed "
                    "lines suppressed -- slice or aggregate instead of "
                    "printing whole sets]"] + lines[-5:]
                lines = kept
            text = "\n".join(lines)
            if len(text) > PRINT_CHAR_CAP:
                text = (text[:PRINT_CHAR_CAP]
                        + f"\n... [printed output clipped at "
                          f"{PRINT_CHAR_CAP:,} chars]")
            out["printed"] = text
        # Named picks: the fid a model-named Row holds, receipted once.
        picked = {}
        for k, v in self.ns.items():
            if isinstance(v, Row) and not k.startswith("__"):
                if self._picked.get(k) != v.fid and v.fid:
                    pk = {"fid": str(v.fid)}
                    if v.name:
                        pk["name"] = str(v.name)
                    picked[k] = pk
                self._picked[k] = v.fid
        if picked:
            out["picked"] = picked
        # Serve the model's own names over the executor's refs.
        mine = dict(self._shown)
        for k, v in self.ns.items():
            if isinstance(v, Rows) and not v.spec and not k.startswith("__"):
                mine[v.ref] = k
        res = out.get("result")
        if isinstance(res, dict) and isinstance(res.get("results"), dict):
            results = dict(res["results"])
            # ONE BLOCK, ONE TABLE: the block's LAST result and everything
            # the model explicitly showed keep their rows; every other
            # envelope folds to its hint under the model's name for it.
            keys = list(results)
            answer = keys[-1] if keys else None
            _a = results.get(answer)
            said = set(str(_a.get("hint") or "").splitlines()
                       if isinstance(_a, dict) else ())
            folded = []
            for k in keys[:-1]:
                if k in self._block_shown:
                    continue
                env = results.pop(k)
                hint = env.get("hint") if isinstance(env, dict) else None
                # Under a name the model can read, else unprefixed; a line
                # the answer already says is dropped, but a named result
                # keeps its own status line.
                name = mine.get(k) or self._labels.get(k)
                keep = [ln for i, ln in enumerate(str(hint or "").splitlines())
                        if (i == 0 and name) or ln not in said]
                if keep:
                    hint = "\n".join(keep)
                    folded.append(f"{name}: {hint}" if name else hint)
            if folded and isinstance(results.get(answer), dict):
                a = dict(results[answer])
                a["hint"] = "\n".join(
                    ([a["hint"]] if a.get("hint") else []) + folded)
                results[answer] = a
            keys, n = [], 0
            for k in results:
                if k in mine:
                    keys.append(mine[k])
                else:
                    n += 1
                    keys.append(f"(unnamed {n})")
            out["result"] = {**res,
                             "results": dict(zip(keys, results.values()))}
        names = sorted(k for k, v in self.ns.items()
                       if isinstance(v, (Rows, Row))
                       and not k.startswith("__"))
        if names:
            out["handles"] = names
        return out


# --------------------------------------------------------------------------
# The liveness ceilings, charged to the block's OWN lines and nothing else
# --------------------------------------------------------------------------
#
# ``sys.monitoring`` delivers LINE events for exactly the block's code
# objects, so the executor beneath the block runs untraced.

_MON = sys.monitoring
_MON_LOCK = threading.Lock()
_MON_TOOL: list = [None]          # the tool id, once claimed
_MON_ACTIVE: dict = {}            # id(code object) -> the MissionPython running it


def _code_objects(code) -> list:
    """A code object and every code object nested in its constants: the
    block's lambdas, comprehensions and defs all count as its own lines."""
    out = [code]
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            out.extend(_code_objects(c))
    return out


def _on_line(code, _line):
    mp = _MON_ACTIVE.get(id(code))
    if mp is not None:
        mp._tick()


def _monitor_tool() -> int:
    with _MON_LOCK:
        if _MON_TOOL[0] is None:
            tid = next(i for i in (3, 4, 5, 2, 1, 0)
                       if _MON.get_tool(i) is None)
            _MON.use_tool_id(tid, "maps-block-ceilings")
            _MON.register_callback(tid, _MON.events.LINE, _on_line)
            _MON_TOOL[0] = tid
        return _MON_TOOL[0]


@contextlib.contextmanager
def _block_ceilings(mp: "MissionPython", compiled):
    """Charge ``mp._tick`` for every line of ``compiled`` while it runs."""
    tid = _monitor_tool()
    codes = _code_objects(compiled)
    with _MON_LOCK:
        for c in codes:
            _MON_ACTIVE[id(c)] = mp
            _MON.set_local_events(tid, c, _MON.events.LINE)
    try:
        yield
    finally:
        with _MON_LOCK:
            for c in codes:
                _MON.set_local_events(tid, c, 0)
                _MON_ACTIVE.pop(id(c), None)


def _block_line() -> int:
    """The failing line of the MODEL's code, off the live traceback."""
    _, _, tb = sys.exc_info()
    line = 0
    for fr in traceback.extract_tb(tb):
        if fr.filename == BLOCK_FILENAME:
            line = fr.lineno or line
    return line
