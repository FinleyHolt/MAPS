"""Real-Python front-end for the mission run channel.

The model writes ordinary Python; this module executes it. Every mission
call (`find`, `go_to`, `look_for`, ...) is a real function in one
persistent per-mission namespace, so variables, arithmetic, f-strings,
loops, comprehensions, `print()` and `len()` behave exactly as the model's
priors say they do -- and an error costs the line it is on, never the
lines that already ran.

Each call builds one step and hands it to ``code_surface._plan_of`` ->
``reasoner_loop._run_plan``, which own the op programs, grounding, serving,
flights and receipts.

The guardrails hold at call time: the moving calls take a thing and never a
coordinate, a whole result set cannot be flown to, and a move inside a loop
is refused up front (18 of 35 model-written loops flew in the body). The
pure deterministic stdlib (``math``, ``itertools``, ...) is loaded in the
namespace.
"""

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

from maps.search.analyze_scene import cardinal_side
from maps.search.analyze_scene.geom_compute import COMPASS_UNIT
from maps.search.analyze_scene.geom_program import (
    _AUTOATTACH_PREFIXES, _STANDING_SIDE_PREFIXES, MODEL_UNSERVED_COLUMNS)
from maps.search.analyze_scene.row_geometry import SurfaceGeometry
from maps.search.withhold import withheld
from maps.util.xy_token import is_xy
from maps.search.parse_compile.code_surface import (
    CodeError, act_needs_a_thing, _fid_like, _plan_of, _split_diagonal,
    bind_row, _refuse_metres_in_within)

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
    """The exec globals: an undefined name that SPELLS a served fid IS
    that fid.

    Every receipt prints fids bare, so the model writes go_to(detected_192).
    The name resolves to a :class:`Row` view of the live row when the
    surface has it, else to the fid string. CPython honors __missing__ on a
    dict-subclass globals for LOAD_NAME and LOAD_GLOBAL.

    A name that misses entirely still answers: a name this mission UNBOUND
    says why (a failed rebind removed it), and a near-miss of something
    visible gets the did-you-mean.
    """

    mp: "MissionPython | None" = None

    def __missing__(self, k):
        mp = self.mp
        if _fid_like(k):
            row = mp.env.row_of_fid(k) if mp and mp.env.row_of_fid else None
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
        if m and mp and mp.env.row_of_fid:
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


# Geometry blobs and internals stay off the Python view: they are served
# coarsely through the real columns (xy, end0_xy, end1_xy). mask_centroid is
# hidden because centroid-to-centroid is the measurement this surface exists
# to make impossible (a road way is 7.0 m from the car parked against it by
# footprint, 408.4 m by centroid). The column diet the grid serves is the
# Row's too.
_ROW_HIDDEN = frozenset({
    "mask", "cells", "centerline", "polygon", "contour", "search_text",
    "world_contour", "metadata", "mask_centroid"}) | MODEL_UNSERVED_COLUMNS


def _norm_key(k: str) -> str:
    """A column spelling, normalised the way the producer normalises labels.

    ``geom_program._sanitize_token`` collapses ``[^a-z0-9]+`` to ``_`` on
    every ref label (``"facing poppy_boulevard"``) while the model says
    "Poppy Boulevard"; running BOTH sides through the same collapse lets it
    spell a column by the name it was given.
    """
    return re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")


# The cardinal cut kwargs, derived from the ONE compass table's declared
# four-subset so a wind can never be spelled here that the geometry does not
# compute (and vice versa).
_CARDINAL_KWARGS = frozenset(f"{w}_of" for w in cardinal_side.CARDINALS)


def _direction_key(word):
    """A compass word -> the sort key that puts the FURTHEST that way first.

    A DIRECTION ORDERS GLOBALLY: "the northernmost pool" is a fact about
    the map frame, so this takes no anchor and reads the rows' own ``xy``
    (+x runs EAST, +y runs NORTH) along the compass table's unit vector.
    Client-side on the rows in hand, like the ``key=`` sort.
    """
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
    of its two cardinal cuts against the same place, which is exact for the
    quadrants of a POINT (112 of the corpus's 150 conjunctive cues)."""
    for n in names:
        n = str(n)
        pair = _split_diagonal(n[:-3]) if n.endswith("_of") else None
        if pair:
            return (f" A diagonal is TWO cuts against the same place: "
                    f"{pair[0]}_of=<place>, {pair[1]}_of=<place> -- which is "
                    f"how you say a corner.")
    return ""


_FILTER_KWARGS = _CARDINAL_KWARGS | {"near", "within", "dwithin", "facing", "direction"}


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
    """One selected row: what ``rows[i]`` binds, AND its served columns.

    A Row is a read-only view of the columns the tables already serve:
    attribute access, mapping access (spaced column names like "distance to
    p0" go through ``r["..."]``; ``r.distance_to_p0`` aliases it),
    ``.get``/``.keys``, ``print(r)``. Geometry blobs stay hidden -- the
    coarse columns carry the geometry.

    ``bind`` is what a slot takes -- the fid, or the [x, y] point for a
    computed row (a crossing, a terminus) that no fid-slot reader knows.
    ``src`` remembers where it was indexed from, so showing it renders
    the ROW (a windowed frame keeps the ordering; a bare dict loses it).
    """

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
        if pred is not None:
            if kw:
                raise CodeError("filter takes a predicate OR "
                                "near=/within=/dwithin=, not both")
            return RowList([self] if pred(self) else [])
        return mp._rows_filter(self, f"{self.fid or 'row'}.filter(...)", **kw)

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
        return mp._rows_order(self, f"{self.fid or 'row'}.order(...)", **kw)

    def _resolve_key(self, k):
        """The served column a Python spelling names, or None.

        ONE resolver for attribute, item and ``.get`` access. A
        REF-LABELLED COLUMN IS SPELLED BY ITS PLACE'S REAL NAME: both sides
        are normalised the producer's way (:func:`_norm_key`), which covers
        all three ref column families (``facing ``, ``distance to ``,
        ``<dir> of ``). An ambiguous normalisation resolves to NOTHING.
        """
        row = object.__getattribute__(self, "_row")
        if not isinstance(k, str) or k in _ROW_HIDDEN:
            return None
        if k in row:
            return k
        hit = [rk for rk in row
               if rk not in _ROW_HIDDEN and _norm_key(rk) == _norm_key(k)]
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
        """Why this key is not here, and what writes it.

        A RELATION COLUMN IS NOT MISSING, IT IS UNASKED-FOR. ``distance to
        <ref>``, ``facing <ref>`` and ``<dir> of <ref>`` exist only where a
        compare named that ref (``geom_program._auto_attach``, whose prefix
        tuple this reads), so the remedy is the relate that writes it.
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
        return [k for k in row if k not in _ROW_HIDDEN
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
            mp = getattr(r, "_mp", None) or object.__getattribute__(r, "_mp")
            if mp is not None:
                return mp
        raise CodeError(
            "these rows are empty, so there is nothing to cut or order")

    def filter(self, pred=None, **kw) -> "RowList | Rows":
        # A PREDICATE FILTER IS NOT A DEAD END: the served cut works on a
        # materialised subset exactly as it does on a handle.
        _refuse_filter_kwargs(kw)
        if pred is None:
            if not kw:
                raise CodeError(
                    "filter needs a predicate or near=/within=/dwithin=: "
                    "rows.filter(lambda r: r.distance(<place>) <= 25), or "
                    "rows.filter(near=<place>, dwithin=25).")
            return self._mp()._rows_filter(list(self), "rows.filter(...)",
                                           **kw)
        if kw:
            raise CodeError("filter takes a predicate OR "
                            "near=/within=/dwithin=, not both")
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
            return self._mp()._rows_order(list(self), "rows.order(...)", **kw)
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
            name = self.mp.name_of(self) or self.ref
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
            name = self.mp.name_of(self) or self.ref
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
        st = {"call": "stratum", "target": None,
              "args": {"of": src.ref, "tier": tier}}
        name = self.mp.name_of(self) or self.ref
        self.mp._dispatch([st], f"{name}.{tier}", {src.ref: src.ref})
        return Rows(st["_ref"], self.mp)

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
        if pred is not None:
            if (near is not None or within is not None or dwithin is not None
                    or facing is not None or direction is not None):
                raise CodeError("filter takes a predicate OR "
                                "near=/within=/dwithin=/facing=, not both")
            if not callable(pred):
                raise CodeError(
                    f"filter's first argument is a predicate and {pred!r} "
                    "is not callable: rows.filter(lambda r: r.distance(x) <= 25)")
            return RowList(r for r in self if pred(r))
        src = self if not self.spec else self.mp._materialize(self)
        name = self.mp.name_of(self) or self.ref
        return self.mp._rows_filter(
            src, f"{name}.filter(...)", within=within, near=near,
            dwithin=dwithin, north_of=north_of, south_of=south_of,
            east_of=east_of, west_of=west_of,
            facing=facing, direction=direction)

    def order(self, key=None, *, nearest=None, color=None,
              direction=None, reverse=False) -> "Rows | RowList":
        """Put the rows in ONE order. A table sorts by one column.

        A DIRECTION ORDERS GLOBALLY: ``direction=`` and ``key=`` order the
        rows IN HAND (``reverse=`` flips a key sort, as ``sorted()`` does);
        ``nearest=`` and ``color=`` re-rank the whole set server-side.

        Relating rows to a place stamps ``facing <place>`` and ``distance
        to <place>``; the signed ``<dir> of <place>`` column rides a
        cardinal CUT against that same place.
        """
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
        name = self.mp.name_of(self) or self.ref
        return self.mp._rows_order(src, f"{name}.order(...)",
                                   nearest=nearest, color=color)

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


def _places(v, who: str) -> tuple:
    """A Python value -> the ``(fid, xy)`` places geometry measures over.

    A SET IS ONE PLACE AND MEANS ITS NEAREST MEMBER. The fid rides so the
    geometry measures the row's footprint, not its point; the point rides
    because a COMPUTED row (a crossing, a terminus) is nowhere on the
    surface and its position is its whole footprint.
    """
    if isinstance(v, Row):
        return ((v.fid, _tuple_xy(v.get("xy"))),)
    if isinstance(v, Rows):
        return tuple((r.get("fid"), _tuple_xy(r.get("xy")))
                     for r in v._rows())
    if is_xy(v):
        return ((None, (float(v[0]), float(v[1]))),)
    if isinstance(v, (list, tuple)):
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


class MissionEnv:
    """The hooks the loop hands this front-end. One per mission.

    ``run_plan(plan, said) -> ack`` executes one plan through the shared
    machinery; ``rows_of_ref(ref)`` reads the executor's carry store;
    ``plan_rn`` is the shared ref counter (the loop and flights advance
    it); ``discourse`` is the mission discourse dict (widening ledger);
    ``minted`` is the loop's flight-minted-handle list, drained here;
    ``say_err(ack, msg)`` formats a refusal the way every surface does;
    ``row_of_fid(fid)`` reads one live-surface row (None when the
    surface holds no such fid -- the bare name then rides as a string).
    """

    def __init__(self, *, run_plan, rows_of_ref, plan_rn, discourse,
                 tied_of_ref=None,
                 minted, say_err, row_of_fid=None):
        self.run_plan = run_plan
        self.rows_of_ref = rows_of_ref
        # Whether a ref holds a TIED view -- the ways of one name, which
        # have no order for an index to mean anything by.
        self.tied_of_ref = tied_of_ref or (lambda _ref: False)
        self.plan_rn = plan_rn
        self.discourse = discourse
        self.minted = minted
        self.say_err = say_err
        self.row_of_fid = row_of_fid


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
        self._picked: dict = {}         # fids the model's Row names hold
        self._unbound: dict = {}        # name -> why this mission removed it
        self._get_noted: set = set()    # .get near-miss keys noted, per block
        # Footprint geometry BY FID off the live surface. The cache is
        # per-block: a flight folds detections in under it, so last block's
        # union would measure a world that no longer stands.
        self._geom = SurfaceGeometry(env.row_of_fid or (lambda fid: None))
        self._measure_calls = 0
        # per-block state
        self._out: dict = {}
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

    def handles(self) -> dict:
        """Model name -> what compile/_site resolve it to."""
        out = {}
        for k, v in self.ns.items():
            if k.startswith("__"):
                continue
            if isinstance(v, Rows) and not v.spec:
                out[k] = v.ref
            elif isinstance(v, Row):
                out[k] = v.bind
            elif isinstance(v, str) and re.fullmatch(r"v\d{1,3}", v) \
                    and k == v:
                out[k] = v              # a staged viewpoint handle
        return out

    # -- the per-call plumbing --------------------------------------------

    def _dispatch(self, steps: list[dict], said: str,
                  extra_handles: dict | None = None) -> dict:
        bound = self.handles()
        if extra_handles:
            bound.update(extra_handles)
        plan, _ = _plan_of(steps, said, self.env.plan_rn[0], bound,
                           self.env.discourse)
        for lbl, ref in (plan.get("_code_shows") or {}).items():
            if isinstance(ref, str):
                self._shown[ref] = lbl
                self._block_shown.add(ref)
        ack = self.env.run_plan(plan, said)
        self._fold(ack)
        self._call_minted = list(self.env.minted)
        for h in self._call_minted:
            self.ns[h] = Rows(h, self)
        del self.env.minted[:]
        if "error" in ack:
            raise _Stop()
        return plan

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

    def _slot(self, v, extra: dict):
        """A Python value -> the arg shape the compiler reads.

        Rows/Row ride as their (registered) name so the served columns
        carry the model's own label; strings and [x, y] points ride
        verbatim -- a bare name grounds through the same plumbing find
        and the same floor it always did.
        """
        if isinstance(v, Rows):
            if v.spec:
                v = self._materialize(v)
            name = self.name_of(v) or v.ref
            extra[name] = v.ref
            return name
        if isinstance(v, Row):
            name = self.name_of(v) or (v.fid or "row")
            extra[name] = v.bind
            return name
        if isinstance(v, (list, tuple)) and not is_xy(v) \
                and any(isinstance(x, (Row, Rows)) for x in v):
            # a LIST of things -- order(nearest=[roadA, roadB]) -- maps
            # element-wise, so a Row rides as its fid, never as an object.
            return [self._slot(x, extra) for x in v]
        return v

    def _materialize(self, sliced: "Rows") -> "Rows":
        """A sliced set becomes a real ref via the identity window, so a
        slice rides every slot a set can (find(near=r1[0:5]) is the union
        of the head, which is what the spelling plainly means)."""
        st = {"call": "window", "target": None,
              "args": {"win_of": sliced.ref, "spec": sliced.spec}}
        self._dispatch([st], f"{sliced.ref}{sliced.spec}",
                       {sliced.ref: sliced.ref})
        return Rows(st["_ref"], self)

    def _step(self, call: str, pos: list, kwargs: dict | None = None,
              act: bool = False) -> dict:
        args = dict(kwargs or {})
        if pos:
            args["_pos"] = pos
        st = {"call": call, "args": args, "target": None}
        if act:
            st["act"] = True
        return st

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
        for _nm in ("nearest", "near"):
            _v = nearest if _nm == "nearest" else near
            if isinstance(_v, (list, tuple)) and len(_v) == 1 \
                    and not is_xy(_v):
                # near=[one thing] is the thing
                if _nm == "nearest":
                    nearest = _v[0]
                else:
                    near = _v[0]
        _refuse_metres_in_within(within)
        extra: dict = {}
        kw = {"nearest": ([self._slot(a, extra) for a in nearest]
                          if isinstance(nearest, (list, tuple))
                          and not is_xy(nearest)
                          else self._slot(nearest, extra)
                          if nearest is not None else None),
              "near": (self._slot(near, extra)
                       if near is not None else None),
              "within": (self._slot(within, extra)
                         if within is not None else None),
              "dwithin": dwithin,
              "color": color, "name": name}
        for d, a in (("north", north_of), ("south", south_of),
                     ("east", east_of), ("west", west_of)):
            kw[f"{d}_of"] = (self._slot(a, extra)
                             if a is not None else None)
        st = self._step("find", [str(query)] if query != "" else [], kw)
        self._dispatch([st], f"find({query!r})", extra)
        self._near_miss_note(st.get("_ref"), query)
        return Rows(st["_ref"], self, label=f"find({query!r})")

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

    def _rows_order(self, items, label, *, nearest=None, color=None):
        """ONE sort key, the way a table sorts. Never two.

        Two keys have to be combined to make one, and every way of
        combining them can rank a row that is second on everything below
        wrong rows. So this refuses rather than fuses, and what does not
        order is a CUT -- .filter(near=..., dwithin=...).
        """
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
        extra: dict = {}
        near = nearest
        if isinstance(near, (list, tuple)) and not is_xy(near):
            # nearest=[A, B] is a SET -- one place, its nearest member --
            # not two keys to average.
            near = [self._slot(a, extra) for a in near]
        elif near is not None:
            near = self._slot(near, extra)
        st = self._step("order", [self._slot(items, extra)],
                        {"nearest": near, "color": color})
        self._dispatch([st], label, extra)
        return Rows(st["_ref"], self)

    def _rows_filter(self, items, label, *, within=None, near=None,
                     dwithin=None, facing=None, direction=None, **cards):
        # Which-arg validation lives in code_surface's filter branch; every
        # caller has already refused a kwarg outside the set.
        _refuse_metres_in_within(within)
        extra: dict = {}
        st = self._step(
            "filter", [self._slot(items, extra)],
            {"within": (self._slot(within, extra)
                        if within is not None else None),
             "near": (self._slot(near, extra)
                      if near is not None else None),
             "dwithin": dwithin,
             # facing= is a PLACE and rides the slot resolver; direction=
             # is the compass WORD and rides verbatim, which is the whole
             # point of keeping the word out of the kwarg name.
             "facing": (self._slot(facing, extra)
                        if facing is not None else None),
             "direction": direction,
             **{d: (self._slot(v, extra) if v is not None else None)
                for d, v in cards.items()}})
        self._dispatch([st], label, extra)
        return Rows(st["_ref"], self)

    def _call_intersection(self, a, b):
        extra: dict = {}
        st = self._step("intersection",
                        [self._slot(a, extra),
                         self._slot(b, extra)])
        self._dispatch([st], "intersection(...)", extra)
        return Rows(st["_ref"], self)

    def _call_ends_of(self, x):
        extra: dict = {}
        st = self._step("ends_of", [self._slot(x, extra)])
        self._dispatch([st], "ends_of(...)", extra)
        return Rows(st["_ref"], self)

    def _call_more_of(self, x):
        if not isinstance(x, Rows) or x.spec:
            raise CodeError("more_of pages a RESULT a find or an ordering "
                            "left")
        name = self.name_of(x) or x.ref
        st = self._step("more_of", [name])
        self._dispatch([st], f"more_of({name})", {name: x.ref})
        return Rows(st["_ref"], self)

    # -- measuring ---------------------------------------------------------

    def _measured(self, call: str) -> None:
        self._measure_calls += 1
        if self._measure_calls > MEASURE_CALL_CEILING:
            raise CodeError(
                f"this block ran past {MEASURE_CALL_CEILING:,} measurements. "
                f"{call}() in a loop inside a loop is one call per PAIR, "
                "which is millions over sets this size. Slice the sets "
                "down, or measure each thing against the WHOLE other set "
                "in one call -- a set is one place and means its nearest "
                "member.")

    def _call_distance(self, a, b) -> float:
        """Metres between two footprints. 0.0 on touch or containment.

        Mask to mask, which is the quantity the ``distance to <ref>``
        column has always carried and the one an ``xy`` cannot reach:
        Poppy Boulevard is 1,198 m long, and the car parked against it is
        7.0 m away where the two served points are 408 m apart. Either
        side may be a set, and a set is ONE place meaning its nearest
        member.
        """
        self._measured("distance")
        return self._geom.distance(_places(a, "distance"),
                                   _places(b, "distance"))

    def _call_look(self, x, *, marks=None):
        extra: dict = {}
        # marks= names the rows whose identity bears on the decision; a
        # named road is drawn as its NAME, everything else outlines and
        # badges.
        _mk = None
        if marks is not None:
            _mk = [self._slot(m, extra)
                   for m in (marks if isinstance(marks, (list, tuple))
                             else [marks])]
        if isinstance(x, Rows) and x.spec:
            win = {"call": "window", "target": "__w0",
                   "args": {"win_of": self.name_of(x) or x.ref,
                            "spec": x.spec}}
            extra[self.name_of(x) or x.ref] = x.ref
            look = self._step("look", ["__w0"], {"marks": _mk})
            self._dispatch([win, look], "look(...)", extra)
            return None
        st = self._step("look", [self._slot(x, extra)],
                        {"marks": _mk})
        self._dispatch([st], "look(...)", extra)
        return None

    def _call_show_map(self, around=None):
        extra: dict = {}
        if isinstance(around, Rows):
            name = self.name_of(around) or around.ref
            raise CodeError(
                f"around= takes ONE place and {name} is a set. Index it "
                f"-- around={name}[0] -- or use a name. (A set IS one "
                "place in near=, within= and nearest=, where it means "
                "the gap to its NEAREST member, but there is no one "
                "point to centre on.)")
        st = self._step(
            "show_map", [],
            {"around": (self._slot(around, extra)
                        if around is not None else None)})
        self._dispatch([st], "show_map(...)", extra)
        return None

    def _call_look_for(self, *classes, at=None):
        # at=, not near=: this names WHERE to segment extra.
        if not classes:
            raise CodeError('look_for takes the classes to watch: '
                            'look_for("house", "pool")')
        extra: dict = {}
        st = self._step(
            "look_for", [str(c) for c in classes],
            {"at": (self._slot(at, extra)
                    if at is not None else None)})
        self._dispatch([st], "look_for(...)", extra)
        return None

    def _call_stand(self, *, at, facing="north"):
        extra: dict = {}
        if isinstance(at, Rows):
            raise CodeError("at= takes ONE place and a set has no one "
                            "point to stand at. Index it -- at=rows[0].")
        st = {"call": "stand", "target": None,
              "args": {"at": self._slot(at, extra),
                       "facing": (self._slot(facing, extra)
                                  if not isinstance(facing, str)
                                  else facing)}}
        self._dispatch([st], "stand(...)", extra)
        staged = (self._out.get("stance") or {}).get("staged") or []
        for s in staged:
            h = s.get("handle")
            if isinstance(h, str):
                self.ns[h] = h
        return staged[-1].get("handle") if staged else None

    def _move(self, call: str, x):
        extra: dict = {}
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
        elif isinstance(x, (list, tuple)):
            # explore's list is a SET of things -- rows, sets, fids -- as
            # one place to cover; any numeric element is the coordinate
            # hatch.
            _things = (bool(x) and not is_xy(x)
                       and all(isinstance(e, (Row, Rows, str)) for e in x))
            _ok_set = call == "explore" and _things
            # A SET OF THINGS IS NOT A COORDINATE: go_to refuses it as a set.
            if _things and not _ok_set:
                raise CodeError(
                    f"{call}() takes ONE thing and that is a set of "
                    f"{len(x)}. Index the one you mean -- {call}(<set>[0]) "
                    "-- or explore(<set>) to cover all of it.")
            if not _ok_set:
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
        st = self._step(call, [self._slot(x, extra)], act=True)
        self._dispatch([st], f"{call}(...)", extra)
        # The arrivals: a flight minted a handle over the rows it added,
        # so `r = explore(x)` binds them.
        minted = getattr(self, "_call_minted", None) or []
        return self.ns.get(minted[-1]) if minted else None

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
            st = {"call": "window", "target": None,
                  "args": {"win_of": name, "spec": value.spec}}
            self._dispatch([st], label, {name: value.ref})
            if st.get("_ref"):
                # Key the envelope by the MODEL's spelling ("r1[0:5]"),
                # not the executor's ref the window compiled onto.
                self._shown[st["_ref"]] = label
        elif isinstance(value, Row) and value.src \
                and not re.fullmatch(r"\w+", label):
            # Render the ROW: a one-row window keeps the ordering
            # attribute a bare dict loses, so the model sees the row it
            # asked for rather than a census of one.
            ref, i = value.src
            st = {"call": "window", "target": None,
                  "args": {"win_of": ref, "spec": f"[{i}:{i + 1}]"}}
            self._dispatch([st], label, {ref: ref})
            if st.get("_ref"):
                self._shown[st["_ref"]] = label
        elif isinstance(value, Row):
            self._out.setdefault("shown", {})[label] = value.bind
        else:
            self._printed.append(f"{label} = {value!r}"
                                 if label != repr(value) else repr(value))

    # -- the block runner --------------------------------------------------

    def run_block(self, code: str) -> dict:
        self._out = out = {"said": code.strip()}
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
            folded = []
            for k in keys[:-1]:
                if k in self._block_shown:
                    continue
                env = results.pop(k)
                if isinstance(env, dict) and env.get("hint"):
                    folded.append(f"{mine.get(k, k)}: {env['hint']}")
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
# objects, so the executor beneath the block runs untraced (a global
# ``sys.settrace`` slowed one find from 1.26 s to 17.2 s). The trace function
# serves a Python without ``sys.monitoring`` (< 3.12): a slow executor beats
# an unbounded one.

_MON = getattr(sys, "monitoring", None)
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
    if _MON is None:
        def _tracer(frame, event, arg):
            if frame.f_code.co_filename != BLOCK_FILENAME:
                return None
            mp._tick()
            return _tracer
        sys.settrace(_tracer)
        try:
            yield
        finally:
            sys.settrace(None)
        return
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
