"""Real-Python front-end for the mission run channel.

The model writes ordinary Python; this module executes it. Every mission
call (`find`, `go_to`, `look_for`, ...) is a real function in one
persistent per-mission namespace, so variables, arithmetic, f-strings,
loops, comprehensions, `print()` and `len()` behave exactly as the model's
priors say they do -- and an error costs the line it is on, never the
lines that already ran.

What executes underneath is unchanged: each call builds the SAME parsed
step the retired parser produced and hands it to ``code_surface._plan_of``
-> ``reasoner_loop._run_plan``, so the op programs, grounding, serving,
flights and receipts are byte-for-byte the ones the DSL arm measured.

The guardrails that are science stay, moved from parse time to call time:
the moving calls take a thing and never a coordinate, a whole result set
cannot be flown, and a move inside a loop is refused up front (measured:
18 of 35 model-written loops flew in the body). Everything else that the
old surface refused because it could not compile it -- print, while, if,
def, slicing, len -- is simply Python here, and the pure deterministic
stdlib (``math``, ``itertools``, ...) is loaded in the namespace: the
top kicked-back family on the first fleets was the model hand-rolling
Euclidean distance ("without math module", its own comment) and mangling
the exponent, block after block.
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
from maps.search.parse_compile.code_surface import (
    CodeError, act_needs_a_thing, _fid_like, _plan_of, _split_diagonal,
    bind_row, _refuse_metres_in_within)

BLOCK_FILENAME = "<mission-block>"

# Liveness ceiling on one block's own lines (harness frames are untraced).
# A block that runs 100k model-written lines is a runaway loop, not a plan.
TRACE_LINE_CEILING = 100_000

# Wall-clock ceiling on ONE block's execution. The line ceiling catches a
# runaway loop; this catches a BOUNDED loop whose calls are each slow -- a
# `for d in rows: find(near=d, dwithin=40)` over the live world state ran ~7
# minutes on a live sortie and froze the whole mission, since the runner
# thread is the one executing the block. Past this the block stops at its
# line with a served error the model can act on, the way a syntax error does.
TRACE_DEADLINE_S = 90.0

_MOVE_CALLS = ("go_to", "explore")
# order/filter are METHODS on a result (rows.order(...), rows.filter(...)):
# a cut and a sort belong to the rows they act on, which is the spelling
# the model reached for on its own. They are served through Rows, not here.
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
    # There is one distance on this surface and it is the footprint gap,
    # so the measurement is spelled with the word the served column
    # already uses ("distance to <ref>").
    "gap": "distance(a, b), or a.distance(b)",
}

_FORBIDDEN_NAMES = frozenset({
    "exec", "eval", "compile", "open", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "input", "exit",
    "quit", "help", "__import__"})

_SAFE_BUILTINS = {n: getattr(builtins, n)
                  for n in ("len", "min", "max", "sorted", "sum", "abs",
                            "round", "range", "enumerate", "zip", "str",
                            "int", "float", "bool", "list", "dict",
                            "tuple", "set", "reversed", "any", "all",
                            "isinstance", "hasattr", "map", "filter",
                            "repr", "divmod", "pow", "type", "dir",
                            "next",
                            # try/except is ordinary Python, and a bare
                            # `except Exception:` died on NameError
                            # (calcity::113#112 b32) -- the standard
                            # exception names are part of the language.
                            "Exception", "ValueError", "KeyError",
                            "TypeError", "IndexError", "AttributeError",
                            "ZeroDivisionError", "RuntimeError",
                            "StopIteration", "ArithmeticError",
                            "LookupError")}

# Pure, deterministic stdlib only: no I/O, no clock, no randomness --
# anything ambient would break the identical-bytes -> identical-trajectory
# property every ledger A/B leans on, and numpy-class C loops would bypass
# the trace-line liveness ceiling. Grown by measured demand (the
# kickback_audit import histogram), never by speculation. `statistics` is
# OUT on the same rule: NormalDist.samples() draws from ambient
# `random.gauss` -- verified two identical blocks, two different results.
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

# The printed channel is capped the way a notebook caps cell output: a
# loop that prints 3,414 rows must not flood the conversation -- one
# episode printed a city-wide result, blew the context, and died with
# zero flights (compaction_infeasible; at least 5 SR=0 siblings).
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

    The model writes go_to(detected_192) because every receipt prints
    fids bare -- 541 NameErrors over 149 episodes on the first pysurface
    fleet, 92% self-repaired but each one a wasted turn. A fid string is
    what every slot takes, so the name resolves to itself -- and when the
    live surface can hand over the row, to a :class:`Row` view of it,
    because the next thing the model does with a bare fid is read it
    (``detected_192.xy``: the 'str' object AttributeError family).
    CPython honors __missing__ on a dict-subclass globals for LOAD_NAME
    and LOAD_GLOBAL, which is what makes this a namespace fact rather
    than a parser feature.

    A name that misses entirely still answers: a name this mission
    UNBOUND says why (a failed rebind removed it), and a near-miss of
    something visible gets the did-you-mean (``show(...)`` was reached
    for 25 times on arm 1; ``show_map`` existed the whole time).
    """

    mp: "MissionPython | None" = None

    def __missing__(self, k):
        from maps.search.parse_compile.code_surface import _fid_like
        mp = self.mp
        if _fid_like(k):
            row = mp.env.row_of_fid(k) if mp and mp.env.row_of_fid else None
            return Row(row, mp=mp) if isinstance(row, dict) else k
        # A plain KeyError is what lets LOAD_GLOBAL fall through to
        # __builtins__ -- a CodeError here would make len()/list()/dict()
        # unreachable (measured in review: `list` did-you-meant `bisect`).
        if k in (dict.get(self, "__builtins__") or {}):
            raise KeyError(k)
        if mp and k in mp._unbound:
            raise CodeError(f"{k} is unbound: {mp._unbound[k]}")
        # A near-miss of a LIVE fid names the correction (go_to(detect_59)
        # with detected_59 printed one block earlier got a bare NameError)
        # -- but ONLY when the prefix itself is a near-miss of a fid
        # family: an ordinary `corner_5` variable typo must fall through
        # to the model's own bound handles, not to a coincidental prior_5.
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
        # A cut and a sort MOVED onto the rows they act on. This is not an
        # alias and calls nothing: the name stays undefined, and the
        # refusal spells where the call went. Without it a model reaching
        # for the old name gets a bare NameError and spends the turn
        # guessing, which measures its rename luck rather than the shape.
        _moved = difflib.get_close_matches(k, tuple(_MOVED_TO_METHOD),
                                           n=1, cutoff=0.8)
        if k in _MOVED_TO_METHOD or _moved:
            k = k if k in _MOVED_TO_METHOD else _moved[0]
            raise CodeError(
                f"there is no {k}() here -- did you mean "
                f"{_MOVED_TO_METHOD[k]}? A cut and a sort are methods on the "
                "rows they act on.")
        # The suggestion pool is what the model plausibly MISSPELLED: the
        # mission calls and its own bound handles -- never the module
        # names, whose close matches are noise ("list" is not "bisect").
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

    A syntax error echoes the offending source line with a caret: the
    grinding families repeat the same slip (``(a)**2`` emitted as
    ``+ **(a)2``) up to 8 blocks in a row when all they are told is
    "invalid syntax".
    """
    try:
        tree = ast.parse(code)
        # Some syntax errors (a repeated keyword argument) are raised by the
        # compiler, not the parser; uncaught they end the episode as infra.
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
                # Same self-shadow class the mission-call guard closed
                # (two episodes looped 32 errors against a shadowed
                # intersection()): one bad bind breaks the module for
                # every remaining turn of the mission.
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

    The repair nudge's detector -- the two-part rule survives the parser
    it used to lean on: real Python parses far more than the DSL did, so
    the library-call requirement is what separates 'wrote code outside
    the tool' from prose that happens to parse.
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
# coarsely through the real columns (xy, end0_xy, end1_xy) and printing a
# row must never dump a polygon into the context.
# mask_centroid rides _MODEL_HIDDEN_ROW_FIELDS on the op path and was not
# hidden here, so a row off the LIVE surface served it. Two points on a row
# is two points to subtract, and centroid-to-centroid is the measurement
# this surface exists to make impossible: Poppy Boulevard way 5 is 7.0 m
# from the car parked against it by footprint and 408.4 m by centroid.
#
# The COLUMN DIET rides here too, for the same reason and by the same
# mechanism. A render cut is not a diet while a second render path exists:
# the grid dropped _GRID_COL_DROP and the Row did not, so `print(row)` --
# which this surface's own docstring teaches -- served every cut column
# straight back. Measured on calcity_conj_93_72fe9987: aspect_ratio 1,315
# occurrences over 49 of 93 episodes, color_conf 1,158 over 43, score 431
# over 45, axis_length_m 192 over 36, name_match 160 over 32, each one a
# column an 8-reader audit found inert or harmful. One rule, both paths.
_ROW_HIDDEN = frozenset({
    "mask", "cells", "centerline", "polygon", "contour", "search_text",
    "world_contour", "metadata", "mask_centroid"}) | MODEL_UNSERVED_COLUMNS


def _norm_key(k: str) -> str:
    """A column spelling, normalised the way the producer normalises labels.

    ``geom_program._sanitize_token`` collapses ``[^a-z0-9]+`` to ``_`` on
    every ref label, so the served key is ``"facing poppy_boulevard"`` while
    the cue -- and therefore the model -- says "Poppy Boulevard". Running
    BOTH sides through the same collapse is what lets the model spell a
    column by the name it was given. Same rule, one place, all three ref
    column families.
    """
    return re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")


# The cardinal cut kwargs, derived from the ONE compass table's declared
# four-subset so a wind can never be spelled here that the geometry does not
# compute (and vice versa).
_CARDINAL_KWARGS = frozenset(f"{w}_of" for w in cardinal_side.CARDINALS)


def _direction_key(word):
    """A compass word -> the sort key that puts the FURTHEST that way first.

    A DIRECTION ORDERS GLOBALLY. "The northernmost pool" is a fact about
    the map frame, not a relation to any object, so this takes no anchor
    and reads the rows' own ``xy`` off the axis line the legend states
    (+x runs EAST, +y runs NORTH). Every relation to a place is a CUT and
    lives on ``filter``.

    The unit vector IS the key: a diagonal is both terms at once, which
    the ONE compass table already carries, and a per-word branch would
    re-derive five hand-written direction tables' worth of the same thing
    (0aea70c7 collapsed exactly that).

    Client-side on the rows in hand, like the ``key=`` sort it sits
    beside. ``nearest=`` is the slot that re-ranks a whole set server-side.
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
    """``order(direction=..., reverse=True)`` is the OPPOSITE direction.

    The model reaches for ``reverse=`` out of ``sorted()`` habit, and on a
    key= sort that is exactly right, so it is served there. On a
    direction= sort it is a second spelling of an ordering the surface
    already has a word for, which is the thing this surface keeps
    refusing. One home so a handle and a list cannot answer differently.
    """
    if not reverse:
        return
    w = str(direction).strip().lower()
    opp = {"north": "south", "south": "north", "east": "west", "west": "east"}
    flip = "".join(opp.get(p, p) for p in (_split_diagonal(w) or (w,)))
    raise CodeError(
        f'order(direction="{w}", reverse=True) is '
        f'order(direction="{flip}") -- say the direction you mean.')


def _diagonal_remedy(names) -> str:
    """The extra sentence a ``<diag>_of=`` mistake earns.

    Listing the four spellings that exist is not enough on its own: the
    model still has to invent the CONJUNCTION. So the refusal writes it,
    because a diagonal cut is exactly its two cardinal cuts against the same
    place -- and on this corpus that is not an approximation. Of the 150
    conjunctive cues, 95 read "the <diag> corner of A and B" and 17 more are
    a diagonal of an intersection or of a street's end: quadrants of a
    POINT, where south-of AND east-of is exact. Only ~14 ask a diagonal of
    an extended feature, which is why cardinal_side offers four winds and
    this names the composition rather than a fifth.
    """
    for n in names:
        n = str(n)
        pair = _split_diagonal(n[:-3]) if n.endswith("_of") else None
        if pair:
            return (f" A diagonal is TWO cuts against the same place: "
                    f"{pair[0]}_of=<place>, {pair[1]}_of=<place> -- which is "
                    f"how you say a corner.")
    return ""


class Row:
    """One selected row: what ``rows[i]`` binds, AND its served columns.

    The model treats a row as data -- ``r.xy``, ``r["class"]``,
    ``r.get("color")``, ``print(r)`` -- because that is what a row is in
    Python (measured on the first pysurface fleet: the thin fid-only Row
    drew 150+ AttributeErrors over 69 episodes). So a Row is a read-only
    view of the columns the tables already serve: attribute access,
    mapping access (spaced column names like "distance to p0" go through
    ``r["..."]``; ``r.distance_to_p0`` aliases it), ``.get``/``.keys``.
    Geometry blobs stay hidden -- the coarse columns carry the geometry.

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

    # ONE row cuts and orders as the one-element set it is. The free calls
    # took a Row in their items slot, and measuring a single candidate
    # against an anchor is the useful half of that: r.order(nearest=poppy)
    # puts a "distance to poppy" column on the row you are about to commit,
    # which is how a constraint stops being something you assumed.
    def distance(self, other) -> float:
        """Metres from this row's footprint to ``other``."""
        mp = object.__getattribute__(self, "_mp")
        return mp._call_distance(self, other)

    def filter(self, pred=None, **kw):
        mp = object.__getattribute__(self, "_mp")
        if "color" in kw:
            raise CodeError(
                "colour never CUTS -- it orders. The same car reads red "
                "0.63 through a tight mask and 0.38 through its full one, "
                "so a colour threshold drops the right row on the width of "
                "its mask. Say rows.order(color=\"red\") and let the "
                "spatial cuts do the dropping.")
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

        ONE resolver for attribute, item and ``.get`` access -- ``.get``
        resolving differently from ``.``/``[]`` was the silent hole: a
        wrong-key ``.get(k, 9999)`` fed its sentinel straight into a real
        filter (22% of arm-1 blocks touch ``.get``).

        A REF-LABELLED COLUMN IS SPELLED BY ITS PLACE'S REAL NAME. Every
        label rides ``geom_program._sanitize_token``, so relating rows to
        Poppy Boulevard writes ``"facing poppy_boulevard"`` -- while the cue
        says "Poppy Boulevard" and the legend's own example said
        ``r["facing Poppy Boulevard"]``, which was a KeyError on the largest
        directional family in the corpus. The model writes the name it was
        given; the producer writes the name it sanitized. Normalising BOTH
        sides the producer's way is what makes those the same lookup, and it
        covers all three ref column families at once (``facing ``,
        ``distance to ``, ``<dir> of ``) because it is the label rule rather
        than a per-prefix alias.

        An ambiguous normalisation resolves to NOTHING. Two served columns
        that normalise alike is a silent bind waiting to happen, and this
        surface would rather refuse and name the columns.
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

    def _miss(self, k) -> str:
        """Why this key is not here, and what writes it.

        A RELATION COLUMN IS NOT MISSING, IT IS UNASKED-FOR. ``distance to
        <ref>``, ``facing <ref>`` and ``<dir> of <ref>`` exist only where a
        compare named that ref, so the model asking for one has not
        misremembered a name -- it has named a place these rows were never
        related to, and the executor would compute the column on request
        (``geom_program._auto_attach``, whose prefix tuple this reads).
        Answering with a truncated list of twelve columns that are not the
        one asked for sent the block looking for a spelling instead of a
        relate: 104 blocks over 77 episodes died this way across six
        500-episode roots, 95 of them on ``distance to <ref>``.
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
        rk = self._resolve_key(k)
        if rk is not None:
            return object.__getattribute__(self, "_row")[rk]
        d = self._derived(k)
        if d is not None:
            return d
        raise AttributeError(self._miss(k))

    def __getitem__(self, k):
        row = object.__getattribute__(self, "_row")
        if isinstance(k, int) and k in (0, -1):
            return self                  # one row indexes as itself
        if isinstance(k, slice):
            return [self]
        rk = self._resolve_key(k)
        if rk is not None:
            return row[rk]
        d = self._derived(k)
        if d is not None:
            return d
        raise KeyError(self._miss(k))

    def get(self, k: str, default=None):
        row = object.__getattribute__(self, "_row")
        rk = self._resolve_key(k)
        if rk is not None:
            return row[rk]
        d = self._derived(k)
        if d is not None:
            return d
        # .get keeps its contract (a true miss returns the default,
        # silently) -- but a miss that NEAR-matches a real column is a
        # misremembered name feeding a sentinel into real logic
        # (`distance to Airway Boulevard` vs the served `along airway`;
        # three burned blocks in one episode). The receipt gets one note
        # per key per block; the return value is untouched.
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

    # Value equality, not identity: a bare fid name mints a FRESH Row
    # view per reference (``__missing__`` caches nothing), so identity
    # comparison made ``detected_192 == detected_192`` False and every
    # model-written dedup/membership test silently wrong. Two views of
    # the same bind are the same place; a Row also equals its own fid
    # string, which is what the receipts print and lists hold.
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

    A predicate keeps an arbitrary subset, and the executor has no ref for
    one -- ``window`` spans a contiguous slice and nothing spans a scatter.
    A list of :class:`Row` already rides every slot (``_slot`` maps a list
    element-wise to fids), so the materialised subset IS first-class; what
    a bare ``list`` loses is the ability to keep chaining. Subclassing
    restores that and costs nothing else: it is still a list everywhere
    else, so a comprehension and a ``.filter`` are interchangeable.
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
        # A PREDICATE FILTER IS NOT A DEAD END. A list of Row rides the
        # items slot element-wise, so the served cut works on a
        # materialised subset exactly as it does on a handle. Refusing the
        # kwarg form here made the first lambda in a block a ONE-WAY DOOR
        # out of every served cut and sort -- measured: calcity::200#198
        # narrowed to cars with a lambda, was refused order(color=), hand
        # scanned fifty rows off printed colour strings and committed 2.5 km
        # away.
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
        # every row ties, so there is no order for [0] or [20:40] to be
        # the [0] or the middle OF. Python refuses the same operation for
        # the same reason, and the model has met that refusal ten thousand
        # times.
        #
        # It is not a rule about care. calcity::284#282 took
        # street_88[0] and swept 1 of 10 segments; calcity::26#26 took
        # road[20:40] and covered a third of Neuralia; calcity::147#145
        # and 26#26 both took pools[0]. Each index READ as a choice and
        # was a coin flip.
        if self._tied():
            name = self.mp.name_of(self) or self.ref
            # NAME THE CALL THAT DOES WORK. A refusal with no exit is
            # routed around: the two episodes that met this one went to a
            # sliceable proxy set and to a raw coordinate, while the
            # episode that met the LOOK refusal -- which names explore --
            # explored all ten segments of its street on the very next
            # block. go_to on a set already names it (_call_go_to).
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

    # THE MATCH TIERS AS ATTRIBUTES. A find scores every row into the
    # retrieval's own tiers, and the agent chooses its stratum by holding
    # it -- r1.exact, explore(r1.exact + r1.tagged) -- instead of by
    # reading a float column the grid does not render. Live views: the
    # full set stays behind this handle, nothing is dropped anywhere.
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
    # Both spellings reach the same two executor ops. The kwarg form is
    # SERVED -- it compiles to a `distance to <anchor>` column and a cut on
    # it, so the measurement is on the table the model reads and the
    # distance is a footprint gap to a set's nearest member. The predicate
    # form is the one the model reached for unprompted
    # (`r1.filter(lambda r: ...)`, calcity::200#198 on the gap arm) and it
    # permits any Python, including centroid arithmetic off an xy -- which
    # is the shape that cost ~18 points. Both exist; the served one is the
    # short one.
    def filter(self, pred=None, *, near=None, within=None, dwithin=None,
               north_of=None, south_of=None, east_of=None, west_of=None,
               facing=None, direction=None,
               **kw) -> "Rows | RowList":
        """Keep the rows that hold. Filters compose; run as many as the
        cue has constraints.

        A DIRECTION FILTERS RELATIVELY and orders globally. Every cut here
        is a relation to a place, so each takes its anchor: ``east_of=``
        keeps what lies that way of it, ``facing=`` with ``direction=``
        keeps what FACES it that way. An anchor may be one place or a set
        of them. Ranking by a direction is a fact about the map rather
        than about any object, so it takes no anchor and lives on
        ``order(direction=...)``.
        """
        if "color" in kw:
            raise CodeError(
                "colour never CUTS -- it orders. The same car reads red "
                "0.63 through a tight mask and 0.38 through its full one, "
                "so a colour threshold drops the right row on the width of "
                "its mask. Say rows.order(color=\"red\") and let the "
                "spatial cuts do the dropping.")
        if kw:
            raise CodeError(
                f"filter takes near=/within=/dwithin= or a predicate; "
                f"{', '.join(sorted(kw))}= is neither."
                + _diagonal_remedy(kw))
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
        # A CARDINAL IS A CUT, on the rows themselves. find(west_of=X) has
        # kept what lies that way of X since before this surface; a set in
        # hand could not ask the same question, so a narrowed candidate set
        # lost the relation entirely.
        return self.mp._rows_filter(
            src, f"{name}.filter(...)", within=within, near=near,
            dwithin=dwithin, north_of=north_of, south_of=south_of,
            east_of=east_of, west_of=west_of,
            facing=facing, direction=direction)

    def order(self, key=None, *, nearest=None, color=None,
              direction=None, reverse=False) -> "Rows | RowList":
        """Put the rows in ONE order. A table sorts by one column.

        A DIRECTION ORDERS GLOBALLY. "Furthest north" is a fact about the
        map frame, not about any object, so ``direction=`` takes no anchor
        -- it ranks on the rows' own ``xy`` off the axis line the legend
        states (+x runs EAST, +y runs NORTH). Relating rows to a PLACE is
        the other question and it is a cut: ``filter(east_of=<place>)``
        keeps what lies that way of it, ``filter(facing=<place>,
        direction=<word>)`` keeps what faces it that way.

        Relating rows to a place stamps ``facing <place>`` and ``distance
        to <place>`` on them; the signed ``<dir> of <place>`` column rides
        a cardinal CUT against that same place.

        This orders the rows IN HAND, like the ``key=`` sort beside it --
        ``nearest=`` is the one that re-ranks the whole set server-side --
        and ``reverse=`` flips it, exactly as ``sorted()`` does.
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
        """Metres to ``other``, this set counting as ONE place.

        The same measurement the free ``gap(a, b)`` makes. Served both
        ways because a surface whose cut and sort are methods reads as one
        where the measurement is too: the arm that served only the free
        call was asked for ``row.gap(other)`` and answered "carries no
        'gap'", which reads as a missing COLUMN.
        """
        return self.mp._call_distance(self, other)

    def __add__(self, other):
        # Set union, the Python way: a plain list of rows to loop and
        # read. Slots still take a real handle; this is for the model's
        # own bookkeeping.
        if isinstance(other, Rows):
            return RowList(list(self) + list(other))
        if isinstance(other, list):
            return RowList(list(self) + other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, list):
            return other + list(self)
        return NotImplemented

    def __repr__(self):
        """Named rows, not an opaque count: `print(rw_end)` served
        `<rw_end: 2 rows>` and the model's next block invented
        `explore(end_0)` off a fid it had never been shown."""
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
            # A TIED VIEW SHOWS ITS SHAPE, the same as the served head. The
            # head deliberately serves no fids and no points, and the repr
            # handed both back: calcity::26#26 wrote print(road), read
            # "prior_5775 [2242.27, 3048.07], ..." off the repr, and flew
            # go_to([2242.27, 3048.07]) -- one way of a 64-way road, reached
            # by a coordinate the surface had refused to serve.
            _nm = str((rows[0] or {}).get("name") or name)
            return f"<{_nm}: {len(rows)} ways>"
        head = ", ".join(_one(r) for r in rows[:3])
        more = f", +{len(rows) - 3} more" if len(rows) > 3 else ""
        return f"<{name}: {len(rows)} row(s): {head}{more}>"


# --------------------------------------------------------------------------
# The measuring calls: footprint geometry between places
# --------------------------------------------------------------------------


# One block's measuring budget. `for h in houses: for p in pools: gap(h, p)`
# is one call per PAIR, and at real set sizes that is millions of mask-to-
# mask scans: the block never finishes and the turn buys nothing. The
# ceiling names the shape back rather than hanging.
MEASURE_CALL_CEILING = 20_000


def _tuple_xy(xy):
    """A served ``xy`` as a hashable pair, or None. The place tuple is the
    geometry cache's key, so a list would make it unhashable."""
    return (float(xy[0]), float(xy[1])) if _is_point(xy) else None


def _places(v, who: str) -> tuple:
    """A Python value -> the ``(fid, xy)`` places geometry measures over.

    ONE coercion for every measuring slot, because a set means the same
    thing in all of them: A SET IS ONE PLACE AND MEANS ITS NEAREST MEMBER.
    That falls out of unioning the members' chains and letting the min-gap
    witness pick, never a ``min()`` over per-member gaps.

    The fid rides even when the row also carries a point: op OUTPUT rows
    strip ``mask``, so a carried row measured off its own point alone would
    silently degrade to a centroid distance. The point rides too, because a
    COMPUTED row (a crossing, a terminus) is nowhere on the surface and its
    position is its whole footprint.
    """
    if isinstance(v, Row):
        return ((v.fid, _tuple_xy(v.get("xy"))),)
    if isinstance(v, Rows):
        return tuple((r.get("fid"), _tuple_xy(r.get("xy")))
                     for r in v._rows())
    if _is_point(v):
        return ((None, (float(v[0]), float(v[1]))),)
    if isinstance(v, (list, tuple)):
        if not v:
            raise CodeError(f"{who}() got an empty list -- there is "
                            "nothing there to measure.")
        return tuple(p for x in v for p in _places(x, who))
    if isinstance(v, str):
        if _fid_like(v):
            return ((v, None),)
        # A NAME HAS NO GEOMETRY until it is retrieved. Grounding one here
        # would be a second grounding path beside find's, answering off a
        # different match than the rows the model is reading.
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
        # show(x) is the model's own display verb (reached 25x on arm 1,
        # steadily since): it IS the bare-expression renderer, so the
        # spelling just works instead of earning a did-you-mean.
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
        ``_call_*``, with signature TypeErrors re-said in the surface's
        own voice. The raw message names ``MissionPython._call_find`` --
        an internal spelling the model then reaches for -- and never
        lists the keywords that DO exist."""
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

    def _slot(self, v, call: str, extra: dict):
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
        if isinstance(v, (list, tuple)) and not _is_point(v) \
                and any(isinstance(x, (Row, Rows)) for x in v):
            # a LIST of things -- order_by(nearest=[roadA, roadB]) -- maps
            # element-wise, so a Row rides as its fid, never as an object.
            return [self._slot(x, call, extra) for x in v]
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
                    and not _is_point(_v):
                # near=[one thing] is the thing
                if _nm == "nearest":
                    nearest = _v[0]
                else:
                    near = _v[0]
        _refuse_metres_in_within(within)
        extra: dict = {}
        kw = {"nearest": ([self._slot(a, "nearest=", extra) for a in nearest]
                          if isinstance(nearest, (list, tuple))
                          and not _is_point(nearest)
                          else self._slot(nearest, "nearest=", extra)
                          if nearest is not None else None),
              "near": (self._slot(near, "near=", extra)
                       if near is not None else None),
              "within": (self._slot(within, "within=", extra)
                         if within is not None else None),
              "dwithin": dwithin,
              "color": color, "name": name}
        for d, a in (("north", north_of), ("south", south_of),
                     ("east", east_of), ("west", west_of)):
            kw[f"{d}_of"] = (self._slot(a, f"{d}_of=", extra)
                             if a is not None else None)
        st = self._step("find", [str(query)] if query != "" else [], kw)
        self._dispatch([st], f"find({query!r})", extra)
        self._near_miss_note(st.get("_ref"), query)
        return Rows(st["_ref"], self, label=f"find({query!r})")

    # The spatial fact behind the prior-anchor miss class (147#145, 26#26):
    # find("pool", near=<street>) re-sorts a global name-match, so 3 pools
    # 2.5 km away read like "the pools near the street". The count trailer
    # says how many detections exist ANYWHERE; this note says none of what
    # was found is NEAR the anchor -- the one sentence the model otherwise
    # has to assemble from three places.
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

        A ranking is one-dimensional: rows come out in one order or they do
        not come out. Two keys have to be combined to make one, and every
        way of combining them loses a row that is second on everything and
        first on nothing -- measured on calcity::200#198, where the target
        sits 6.4 m off the boulevard and 19.9 m from a house and FIVE
        combiners (min-max sum, z-score sum, rank sum, product, distances
        only) each rank a wrong car above it. The fused form was also
        never once written: 587 order_by sites across three flown arms
        passed two kwargs zero times. So this refuses rather than fuses,
        and what does not order is a CUT -- .filter(near=..., dwithin=...).
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
        if isinstance(near, (list, tuple)) and not _is_point(near):
            # nearest=[A, B] is a SET -- one place, its nearest member --
            # not two keys to average.
            near = [self._slot(a, "nearest=", extra) for a in near]
        elif near is not None:
            near = self._slot(near, "nearest=", extra)
        st = self._step("order", [self._slot(items, "order", extra)],
                        {"nearest": near, "color": color})
        self._dispatch([st], label, extra)
        return Rows(st["_ref"], self)

    def _rows_filter(self, items, label, *, within=None, near=None,
                     dwithin=None, facing=None, direction=None, **cards):
        # WHICH-ARG validation lives in ONE place -- ``code_surface``'s
        # filter branch -- so the two refusals cannot drift apart. This
        # passes the slots through and the compiler says what is missing.
        # The one check that CANNOT wait for the compiler is the swapped
        # within=/dwithin= pair: _slot() turns a bare 25 into a slot dict,
        # so by the time the compiler sees it the type evidence is gone.
        #
        # A KWARG OUTSIDE THE SET IS REFUSED, NEVER IGNORED -- code_surface's
        # own stated law, and this is the one funnel every filter path
        # reaches (Rows.filter validates its own **kw, but Row.filter and
        # RowList.filter forward theirs straight here). An unknown name used
        # to land in ``cards`` and be dropped by the compiler's _CARDINALS
        # comprehension, so `rows.filter(southeast_of=X)` cut NOTHING and
        # said nothing: the exact shape of the `toward=` kwarg that was
        # built, accepted, compiled to nothing, and had to be deleted.
        bad = sorted(k for k in cards if k not in _CARDINAL_KWARGS)
        if bad:
            raise CodeError(
                f"filter takes near=/within=/dwithin=/facing=, "
                f"{'/'.join(sorted(_CARDINAL_KWARGS))} or a predicate; "
                f"{'=, '.join(bad)}= is none of them."
                + _diagonal_remedy(bad))
        _refuse_metres_in_within(within)
        # ``within`` is a plain number and rides verbatim.
        extra: dict = {}
        st = self._step(
            "filter", [self._slot(items, "filter", extra)],
            {"within": (self._slot(within, "within=", extra)
                        if within is not None else None),
             "near": (self._slot(near, "near=", extra)
                      if near is not None else None),
             "dwithin": dwithin,
             # facing= is a PLACE and rides the slot resolver; direction=
             # is the compass WORD and rides verbatim, which is the whole
             # point of keeping the word out of the kwarg name.
             "facing": (self._slot(facing, "facing=", extra)
                        if facing is not None else None),
             "direction": direction,
             **{d: (self._slot(v, f"{d}=", extra) if v is not None else None)
                for d, v in cards.items()}})
        self._dispatch([st], label, extra)
        return Rows(st["_ref"], self)

    def _call_intersection(self, a, b):
        extra: dict = {}
        st = self._step("intersection",
                        [self._slot(a, "intersection", extra),
                         self._slot(b, "intersection", extra)])
        self._dispatch([st], "intersection(...)", extra)
        return Rows(st["_ref"], self)

    def _call_ends_of(self, x):
        extra: dict = {}
        st = self._step("ends_of", [self._slot(x, "ends_of", extra)])
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
        # WHAT ELSE IS IN THE PICTURE. marks= names the rows whose identity
        # bears on the decision -- the road a candidate is supposed to be
        # on, the thing it is supposed to be beside. A named road is drawn
        # as its NAME and takes no badge digit; everything else outlines and
        # badges. Naming them is the model stating what it is looking FOR.
        _mk = None
        if marks is not None:
            _mk = [self._slot(m, "marks=", extra)
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
        st = self._step("look", [self._slot(x, "look", extra)],
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
            {"around": (self._slot(around, "around=", extra)
                        if around is not None else None)})
        self._dispatch([st], "show_map(...)", extra)
        return None

    def _call_look_for(self, *classes, at=None):
        # at=, not near=: this names WHERE to segment extra, which is a
        # third job for a word that already meant cut and order.
        if not classes:
            raise CodeError('look_for takes the classes to watch: '
                            'look_for("house", "pool")')
        extra: dict = {}
        st = self._step(
            "look_for", [str(c) for c in classes],
            {"at": (self._slot(at, "at=", extra)
                    if at is not None else None)})
        self._dispatch([st], "look_for(...)", extra)
        return None

    def _call_stand(self, *, at, facing="north"):
        extra: dict = {}
        if isinstance(at, Rows):
            raise CodeError("at= takes ONE place and a set has no one "
                            "point to stand at. Index it -- at=rows[0].")
        st = {"call": "stand", "target": None,
              "args": {"at": self._slot(at, "at=", extra),
                       "facing": (self._slot(facing, "facing=", extra)
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
            # A SET is one place for explore -- the pass covers its
            # union. go_to still lands on exactly one thing.
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
        elif isinstance(x, (list, tuple)):
            # explore's list is a SET of things -- rows, sets, fids -- as
            # one place to cover; any numeric element is the coordinate
            # hatch, refused in every spelling.
            _things = (bool(x) and not _is_point(x)
                       and all(isinstance(e, (Row, Rows, str)) for e in x))
            _ok_set = call == "explore" and _things
            # A SET OF THINGS IS NOT A COORDINATE. go_to takes one thing, so
            # a list is still refused -- but saying "not a coordinate" about
            # a list of rows names the wrong defect and the model has to
            # guess. It costs a turn: canary 139#137 wrote go_to(lot68) over
            # a one-row predicate result and only then found go_to(lot68[0]).
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
            # move wire resolves one to its stand point, which is the
            # coordinate hatch again, two hops long.
            raise CodeError(
                f"{x} is a viewpoint you adopted, not a thing on the "
                f"map, and {call}() takes a thing. A stance is for "
                "looking -- its sides ride every find row. To move, "
                f"{call}() a row: r1[0], or a fid off one.")
        st = self._step(call, [self._slot(x, call, extra)], act=True)
        self._dispatch([st], f"{call}(...)", extra)
        # The arrivals: a flight minted a handle over the rows it added,
        # so `r = explore(x)` binds them (45 NoneType errors on the first
        # fleet came from exactly this expectation).
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
                self._block_shown.add(st["_ref"])
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
                self._block_shown.add(st["_ref"])
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
        side raises -- and a stale pick that stays bound is a wrong commit
        (measured: `r3 = r2[99]` failed, the old r3 stayed live, and the
        next go_to(r3) flew the previous block's row). The model was told
        the bind failed, so the name must be gone, not stale.
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
            # ONE BLOCK, ONE TABLE -- compact_results' answer law at block
            # scope: the block's LAST result keeps its rows, everything the
            # model explicitly showed keeps its rows, and every other
            # envelope folds to its hint, attributed to the model's name
            # for it. The facts survive; the re-render is defunded.
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


def _is_point(v) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in v))


# --------------------------------------------------------------------------
# The liveness ceilings, charged to the block's OWN lines and nothing else
# --------------------------------------------------------------------------
#
# ``sys.monitoring`` delivers LINE events for exactly the block's code
# objects. A ``sys.settrace`` function did this job before, and it is
# GLOBAL: CPython hands it a call event for every Python function the
# executor runs beneath the block and keeps the whole interpreter
# de-specialised while it is installed. Measured on the Cal City surface
# (7,205 priors + 40 folds): one ``find("building", near=d, dwithin=40)``
# ran 1.26 s with no trace function and 17.2 s with one -- the 7-minute
# sortie freeze was six of those, and every flight's fold and surface
# rebuild paid the same tax. A local event costs the executor nothing.
#
# The trace function stays for a Python without ``sys.monitoring`` (< 3.12):
# the ceilings are liveness, and a slow executor beats an unbounded one.

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
