"""The values mission calls hand back: a row (:class:`Row`), a result set
(:class:`Rows`) and a list of rows the model narrowed in Python
(:class:`RowList`), with the one ``filter`` and ``order`` they share."""

from __future__ import annotations

import ast
import difflib
import functools
import inspect
import re
import sys
from typing import TYPE_CHECKING

from maps.library.block_scan import _LIBRARY_CALLS, _shadows
from maps.library.op_compiler import (
    CARDINAL_ORDER, CodeError, bind_row, fid_like, refuse_metres_in_within,
    split_diagonal)
from maps.query.result_tables import (
    AUTOATTACH_PREFIXES, MODEL_UNSERVED_COLUMNS, STANDING_SIDE_PREFIXES)
from maps.query.op_io import empty_window_refusal, sanitize_token, window_span
from maps.query.spatial_math import COMPASS_UNIT
from maps.xy_token import is_xy

if TYPE_CHECKING:
    from maps.library.calls import Library


# What a set the model has no name for is called where a label must be said
# (Names.spelling): never the executor's rN, which it never bound.
UNNAMED = "(unnamed)"

# The cardinal cut kwargs, derived from the compiler's (the geometry's) four.
_CARDINAL_KWARGS = frozenset(f"{w}_of" for w in CARDINAL_ORDER)


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
    flip = "".join(opp.get(p, p) for p in (split_diagonal(w) or (w,)))
    raise CodeError(
        f'order(direction="{w}", reverse=True) is '
        f'order(direction="{flip}") -- say the direction you mean.')


def _kwarg_remedy(names) -> str:
    """The extra sentence a refused filter keyword earns: ``nearest=`` is
    the order it names; a ``<diag>_of=`` is the conjunction of its two
    cardinal cuts against the same place."""
    if "nearest" in names:
        return (" nearest= ORDERS and drops nothing: rows.order(nearest=<place>);"
                " to keep only what is close, rows.filter(near=<place>,"
                " dwithin=<metres>).")
    for n in names:
        n = str(n)
        pair = split_diagonal(n[:-3]) if n.endswith("_of") else None
        if pair:
            return (f" A diagonal is TWO cuts against the same place: "
                    f"{pair[0]}_of=<place>, {pair[1]}_of=<place> -- which is "
                    f"how you say a corner.")
    return ""


_FILTER_KWARGS = _CARDINAL_KWARGS | {"near", "within", "dwithin", "facing", "direction"}
_FILTER_SAY = ("near=/within=/dwithin=/facing= with direction=, "
               + "/".join(f"{k}=" for k in sorted(_CARDINAL_KWARGS)))

_MISSING = object()


def _said(fn, name: str | None = None):
    """``fn`` as the model calls it, ``name`` (a method: its own name): a
    TypeError binding its arguments is re-said without the internal or
    class name and counting no self, with the keywords that do exist; a
    TypeError raised inside it rides as it is. The signature stays
    ``fn``'s; it prints as ``name``, with no class."""
    name = name or fn.__name__
    params = [p for n, p in inspect.signature(fn).parameters.items()
              if n != "self"]
    kws = ", ".join(f"{p.name}=" for p in params
                    if p.kind is p.KEYWORD_ONLY or p.default is not p.empty) or "none"
    n_pos = sum(p.kind is p.POSITIONAL_OR_KEYWORD for p in params)
    takes = (f"{name}() takes {n_pos or 'no'} positional "
             f"argument{'' if n_pos == 1 else 's'}")

    @functools.wraps(fn)
    def call(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except TypeError as e:
            if e.__traceback__.tb_next is not None:
                raise               # raised inside fn, not binding to it
            msg = str(e).replace(f"{fn.__qualname__}()", f"{name}()")
            if re.search(r"takes .*positional arguments? but \d+", msg):
                msg = takes
            raise CodeError(f"{msg}. {name} keywords: {kws}") from None
    call.__name__ = call.__qualname__ = name
    return call


class _Cuts:
    """``filter`` and ``order``, one body and one refusal text for a row, a
    list of rows and a result set. A predicate, a key or a compass word needs
    no server and answers on the rows in hand; a served cut or order runs on
    the mission over ``_target()``."""

    __slots__ = ()

    def __getattr__(self, k: str):
        # Reached only when no method or slot has the name: a SET read as a
        # row (``r1.xy``) or a keyword spelled as a method (``r1.nearest``).
        if k.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {k!r}")
        if k in _LIBRARY_CALLS:
            raise AttributeError(
                f"{k}() is a call, not a method: call it on its own, {k}(...)")
        if isinstance(self, Rows):
            name = self._mp.names.spelling(self)
        else:
            mp = self._mission()
            name = mp and mp.names.name_of(self)
        # A remedy names the set as the model does, else binds it first.
        said, then = (name, "") if name else ("s", "bind it first (s = ...); ")
        kw = ("order" if k in ("nearest", "color", "direction")
              else "filter" if k in _FILTER_KWARGS else None)
        if kw:
            raise AttributeError(
                f"{k} is a keyword, not a method: {then}{said}.{kw}({k}=...)")
        one = _one_row(self, said)
        if one is None:
            whose = "there is no row to read it off."
        elif k in ("keys", "get"):
            whose = (f"{then}{k}() is a row's own: {one}.{k}"
                     f"({'<column>' if k == 'get' else ''})")
        else:
            whose = _column_read(self, k, f"{then}a column is read off one "
                                          f"row: {one}.{k}")
        raise AttributeError(
            f"{name or 'that'} is a set of {len(self)} row(s) and carries no "
            f"{k!r}; {whose}")

    def _not_set_algebra(self, other):
        raise CodeError(
            "rows are not a python set: a + b adds two sets into a list of rows, and "
            "a.filter(lambda r: r in b) keeps the rows of a that b holds")

    __or__ = __ror__ = __and__ = __rand__ = __sub__ = __rsub__ = _not_set_algebra

    def __call__(self, *args, **kwargs):
        # A set or a row is no call; a name bound over a mission call says
        # so. A builtin that calls it (sorted(key=s)) is not its name; a
        # subscript called (cars[0]()) is.
        mp = self._mission()
        node = mp and mp.names.running_node()
        f = node.func if isinstance(node, ast.Call) else None
        # A subscript is the set's own spelling only when called bare: a
        # builtin reached through one passes its arguments.
        name = ((ast.get_source_segment(mp.names.code, f) or ast.unparse(f))
                if isinstance(f, ast.Subscript) and not node.args and not node.keywords
                else f.id if isinstance(f, ast.Name) and dict.get(mp.ns, f.id) is self else
                (mp.names.spelling(self) if isinstance(self, Rows)
                 else mp and mp.names.name_of(self)) or UNNAMED)
        what = "a row" if isinstance(self, Row) else f"a set of {len(self)} row(s)"
        raise TypeError(f"{name} is {what}, not a call"
                        + (f": {_shadows(name)}" if name in _LIBRARY_CALLS else "."))

    def _mission(self):
        """The library of the mission the rows came from; rows that carry
        none (an empty list) are read by a block (two frames up) running in
        the mission's namespace."""
        return (self._target()[0]
                or getattr(sys._getframe(2).f_globals, "mp", None))

    def _in_hand(self):
        return self

    def _sorted(self, key, reverse: bool) -> "RowList":
        return RowList(sorted(self._in_hand(), key=key, reverse=reverse))

    @_said
    def filter(self, pred=None, *, near=None, within=None, dwithin=None,
               north_of=None, south_of=None, east_of=None, west_of=None,
               facing=None, direction=None, **kw):
        """Keep the rows that hold; filters compose, and every cut is a
        relation to its anchor (one place or a set)."""
        # A kwarg outside the set is refused, never ignored, and a colour cut
        # gets its own reason.
        if "color" in kw:
            raise CodeError(
                "colour never CUTS -- it orders. The same car reads red "
                "0.63 through a tight mask and 0.38 through its full one, "
                "so a colour threshold drops the right row on the width of "
                "its mask. Say rows.order(color=\"red\") and let the "
                "spatial cuts do the dropping.")
        if kw:
            bad = sorted(kw)
            raise CodeError(
                f"filter takes {_FILTER_SAY} or a predicate; "
                f"{'=, '.join(bad)}= is none of them."
                + _kwarg_remedy(bad))
        refuse_metres_in_within(within)
        kw = {k: v for k, v in dict(
            near=near, within=within, dwithin=dwithin, north_of=north_of,
            south_of=south_of, east_of=east_of, west_of=west_of,
            facing=facing, direction=direction).items() if v is not None}
        if pred is None:
            mp, src = self._target()
            return RowList() if mp is None else mp.rows_filter(src, **kw)
        if kw:
            raise CodeError(
                f"filter takes a predicate OR {_FILTER_SAY}, not both")
        if isinstance(pred, _Cuts) or not callable(pred):
            raise CodeError(
                f"filter's first argument is a predicate and {pred!r} "
                "is not callable: rows.filter(lambda r: r.distance(x) <= 25)")
        return RowList(r for r in self._in_hand() if pred(r))

    @_said
    def order(self, key=None, *, nearest=None, color=None, direction=None,
              reverse=False, **kw):
        """Put the rows in one order; a table sorts by one column.
        ``direction=`` and ``key=`` sort the rows in hand (``reverse=`` flips a
        key sort, as ``sorted()`` does); ``nearest=`` and ``color=`` re-rank
        the whole set server-side, and relating to a place stamps ``facing
        <place>`` and ``distance to <place>``."""
        if kw:
            raise CodeError(
                "order takes key=/nearest=/color=/direction=/reverse=; "
                f"{'=, '.join(sorted(kw))}= is none of them.")
        if key is None and direction is None:
            if reverse and (nearest is not None or color is not None):
                raise CodeError(
                    "reverse= flips a key sort; nearest= and color= are "
                    "served orders with one direction. Farthest first is a "
                    "key sort: rows.order(lambda r: r.distance(<place>), "
                    "reverse=True).")
            mp, src = self._target()
            return (RowList() if mp is None
                    else mp.rows_order(src, nearest=nearest, color=color))
        if key is not None and direction is not None:
            raise CodeError(
                'order takes a key OR direction=, not both: '
                'direction="north" IS the key.')
        if nearest is not None or color is not None:
            raise CodeError("order takes a key OR nearest=/color=, not both")
        if direction is not None:
            _refuse_reversed_direction(direction, reverse)
            return self._sorted(_direction_key(direction), False)
        if isinstance(key, _Cuts) or not callable(key):
            raise CodeError(
                f"order's first argument is a key function and {key!r} "
                "is not callable: rows.order(lambda r: r.distance(x))")
        return self._sorted(key, bool(reverse))


class Row(_Cuts):
    """One selected row: a read-only view of its served columns (attributes,
    ``r["spaced name"]``, ``.get``/``.keys``, ``print``), geometry blobs
    hidden. ``_bind`` is what a slot takes (the fid, or [x, y] for a computed
    row); ``_src`` is the (ref, index) it was indexed from, so showing it
    renders the row."""

    __slots__ = ("_bind", "fid", "name", "_src", "_row", "_mp")

    def __init__(self, row: dict, src: tuple | None = None, mp=None):
        object.__setattr__(self, "_row", dict(row))
        # The mission's Library, or None for a row made outside one. The
        # slot keeps its name: a block's dir() of a row serves it.
        object.__setattr__(self, "_mp", mp)
        self._bind = bind_row(row)
        self.fid = row.get("fid")
        self.name = row.get("name")
        self._src = src                       # (ref, index) or None

    @_said
    def distance(self, other) -> float:
        """Metres from this row's footprint to ``other``."""
        mp = object.__getattribute__(self, "_mp")
        return mp.distance(self, other)

    # ONE row cuts and orders as the one-element set it is:
    # r.order(nearest=poppy) puts a "distance to poppy" column on the row
    # you are about to commit.
    def _in_hand(self):
        return (self,)

    def _sorted(self, key, reverse: bool) -> "RowList":
        return RowList([self])          # one row is already in order

    def _target(self):
        return object.__getattribute__(self, "_mp"), self

    def _resolve_key(self, k):
        """The served column a Python spelling names, or None: the exact key,
        else the one column equal under ``sanitize_token``, the collapse ref
        labels take ("Poppy Boulevard" names "facing poppy_boulevard"; an
        ambiguous match is None). Attribute, item and ``.get`` access all
        resolve here."""
        row = object.__getattribute__(self, "_row")
        if not isinstance(k, str) or k in MODEL_UNSERVED_COLUMNS:
            return None
        if k in row:
            return k
        hit = [rk for rk in row
               if rk not in MODEL_UNSERVED_COLUMNS
               and sanitize_token(rk) == sanitize_token(k)]
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
            v = object.__getattribute__(self, "_row")[rk]
            return _Tags(v) if rk == "tags" and isinstance(v, dict) else v
        d = self._derived(k)
        return _MISSING if d is None else d

    def _miss(self, k) -> str:
        """Why this key is not here and what writes it: a relation column
        exists only where a compare named its ref
        (``AUTOATTACH_PREFIXES``), so the remedy is the relate.
        """
        row = object.__getattribute__(self, "_row")
        pre = next((p for p in AUTOATTACH_PREFIXES
                    if isinstance(k, str) and k.startswith(p)), None)
        generic = (f"row {row.get('fid')} carries no {k!r}; its columns "
                   f"are {sorted(self.keys())[:12]}")
        if pre is None or pre == "color_":
            # a colour column is a PALETTE word, not a relate; the
            # off-palette word has its own loud error at the op layer
            return generic
        place = k[len(pre):]
        if pre in STANDING_SIDE_PREFIXES:
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
            return RowList([self][k])
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
                mp.note(
                    f".get({k!r}) matched no column and returned the "
                    f"default; the near column is {close[0]!r}")
        return default

    def keys(self) -> list:
        row = object.__getattribute__(self, "_row")
        return [k for k in row if k not in MODEL_UNSERVED_COLUMNS
                and not k.startswith("_")]

    # Value equality: a bare fid name mints a FRESH Row view per reference,
    # and two views of the same bind are the same place. A Row also equals
    # its own fid string.
    def _eq_key(self):
        b = self._bind
        return tuple(b) if isinstance(b, list) else b

    def __eq__(self, other):
        if isinstance(other, Row):
            return self._eq_key() == other._eq_key()
        if isinstance(other, str):
            return other == self.fid or other == self._bind
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


class _Tags(dict):
    """A row's tags, served as ``k=v; k=v`` text: a slice
    of it is refused with the reads it does take."""

    def __getitem__(self, k):
        if isinstance(k, slice):
            raise CodeError(
                "tags is a dict of OSM tags, not text: r.tags[\"building\"], "
                "r.tags.get(\"amenity\"), or str(r.tags) for its text.")
        return dict.__getitem__(self, k)


# type(r.tags), its repr and every error naming its type read as a dict's
_Tags.__name__ = _Tags.__qualname__ = "dict"
_Tags.__module__ = "builtins"


class RowList(_Cuts, list):
    """Rows the model narrowed in PYTHON (an arbitrary subset no executor
    ref spans): a plain list of :class:`Row` that every slot takes
    element-wise, kept a subclass only so ``.filter``/``.order`` chain."""

    __slots__ = ()

    def _target(self):
        """The mission the rows came from, and the rows; ``(None, None)``
        for no rows, which cut and order to no rows. A list carries no ref of
        its own, but every Row in it carries the surface it was minted on, so
        a predicate filter is no dead end: the served cut works on the
        materialised subset as it does on a handle."""
        for r in self:
            mp = object.__getattribute__(r, "_mp") if isinstance(r, Row) else None
            if mp is not None:
                return mp, list(self)
        return None, None

    def __getitem__(self, i):
        try:
            v = list.__getitem__(self, i)
            return RowList(v) if isinstance(i, slice) else v
        except (IndexError, TypeError) as e:
            past_end = isinstance(e, IndexError)
        mp = self._mission()
        name = (mp and (mp.names.name_of(self) or mp.names.call_arg(None))) or UNNAMED
        if past_end:
            raise CodeError(f"{name}[{i}] did not bind: {name} holds "
                            f"{len(self)} row(s)."
                            + ("" if self else " There is no row to index."))
        raise _not_an_int(name, i, self)


class Rows(_Cuts):
    """A result set behind an executor ref: an index gives a :class:`Row`,
    a slice a window over the same ref, ``len()`` the rows it holds now."""

    __slots__ = ("_ref", "_spec", "_mp", "_label")

    def __init__(self, ref: str, mp: "Library", spec: str = "",
                 label: str = ""):
        self._ref = ref
        self._spec = spec
        self._mp = mp
        self._label = label          # the call that made it, for refusals

    def _rows(self) -> list[dict]:
        rows = self._mp.env.rows_of_ref(self._ref)
        if self._spec:
            lo, hi = window_span(self._spec)
            rows = rows[lo:hi]
        return rows

    def _lo(self) -> int:
        """Where this window starts in the whole ref: a row's ``_src`` index
        is into the ref, not the window."""
        return window_span(self._spec)[0] if self._spec else 0

    def __len__(self):
        return len(self._rows())

    def _mission(self):
        return self._mp

    def _tied(self) -> bool:
        return bool(not self._spec and self._mp.env.tied_of_ref(self._ref))

    def _refuse_tied(self, name: str, what: str):
        """The refusal of an index or a page on a tied view; it names the
        call that does the work."""
        raise CodeError(
            f"{name} is a tied view of {len(self._rows())} rows and {what}: "
            "they all match equally, so no row is its first and no span is "
            f"its middle. {name}.order(nearest=<place>) ranks them so [0] is "
            f"the nearest way, a slot like near={name} relates to it as one "
            f"place, and explore({name}) covers the whole set.")

    def __getitem__(self, i):
        # A TIED VIEW IS NOT SUBSCRIPTABLE, the way ``dict.keys()`` is not:
        # every row ties, so an index would read as a choice and be a coin
        # flip.
        if self._tied():
            self._refuse_tied(self._mp.names.said_name(self, None),
                              "is not subscriptable")
        if isinstance(i, slice):
            lo = int(i.start or 0)
            if i.stop is None or i.step not in (None, 1) or lo < 0:
                raise CodeError("a window is [lo:hi] with both ends")
            hi = int(i.stop)
            if self._spec:
                # a slice of a slice composes into absolute indices
                p_lo, p_hi = window_span(self._spec)
                held = range(len(self._mp.env.rows_of_ref(self._ref)))[p_lo:p_hi]
                s = held[lo:hi]
                if held and not s:
                    raise CodeError(
                        empty_window_refusal(f"[{lo}:{hi}]", len(held)))
                lo, hi = s.start, s.stop
            return Rows(self._ref, self._mp, f"[{lo}:{hi}]")
        if not isinstance(i, int):
            raise _not_an_int(self._mp.names.said_name(self, None), i, self)
        rows = self._rows()
        if not (-len(rows) <= i < len(rows)):
            name = self._mp.names.said_name(self, None)
            raise CodeError(
                f"{name}[{i}] did not bind: {name} served {len(rows)} "
                "row(s) here. " + ("Show it first, then index it." if rows
                                   else "There is no row to index."))
        return Row(rows[i], src=(self._ref, self._lo() + i % len(rows)),
                   mp=self._mp)

    def __iter__(self):
        lo = self._lo()
        return (Row(r, src=(self._ref, lo + i), mp=self._mp)
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
        src = self if not self._spec else self._mp._materialize(self)
        p = self._mp._program(src._ref)
        name = self._mp.names.spelling(self)
        return Rows(self._mp._analyze(
            p, p.add("stratum", {"items": src._ref, "tier": tier}),
            shown=name and f"{name}.{tier}"), self._mp)

    # -- cut and order: many filters, ONE ordering ------------------------
    def _target(self):
        return self._mp, self if not self._spec else self._mp._materialize(self)

    @_said
    def distance(self, other) -> float:
        """Metres to ``other``, this set counting as ONE place: the free
        ``distance(a, b)`` as a method."""
        return self._mp.distance(self, other)

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
        name = self._mp.names.spelling(self) or UNNAMED
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


def _tuple_xy(xy):
    """A served ``xy`` as a hashable pair, or None. The place tuple is the
    geometry cache's key, so a list would make it unhashable."""
    return (float(xy[0]), float(xy[1])) if is_xy(xy) else None


# type() of a row or a set names the class, not the module serving it
Row.__module__ = RowList.__module__ = Rows.__module__ = "builtins"


def _index_it(name: str | None) -> str:
    """The remedy's opening for a set refused where one place goes: a set
    the model has no name for is bound first, as ``s``."""
    return "Index it" if name else "Bind it first (s = ...), then index it"


def _column_read(x, k: str, read: str) -> str:
    """A remedy for column ``k`` asked of the set ``x``: ``read``, the read
    off one row, when its first row carries ``k`` (or it holds no row), else
    that row's columns."""
    first = next((r for r in x if isinstance(r, Row)), None)
    return read if first is None or first._column(k) is not _MISSING else first._miss(k)


def _not_an_int(name: str, i, x) -> CodeError:
    """The refusal of the set ``x`` indexed by a non-int; a str is a
    column, read off one row."""
    return CodeError(
        f"{name}[{i!r}] did not bind: {name} is a set of rows, indexed by an "
        "int position" + ("; " + _column_read(
            x, i, f"a column is read off one row: {name}[0][{i!r}].")
            if isinstance(i, str) and len(x) else "."))


def _one_row(x, said: str) -> str | None:
    """How a remedy spells one row of the set ``x`` the model calls
    ``said``: its first row, a tied view's nearest way, None when it holds
    no row."""
    if not len(x):
        return None
    if isinstance(x, Rows) and x._tied():
        return f"{said}.order(nearest=<place>)[0]"
    return f"{said}[0]"


def _set_as_list(v) -> list:
    """A Python SET of things is the list of them, in a fixed order."""
    return sorted(v, key=lambda x: str(x._bind) if isinstance(x, Row)
                  else repr(x))


def _places(v, who: str) -> tuple:
    """A Python value -> the ``(fid, xy)`` places geometry measures over. A set
    is one place meaning its nearest member; the xy rides for a computed row (a
    crossing, a terminus), whose point is its whole footprint."""
    if isinstance(v, Row):
        return ((v.fid, _tuple_xy(v.get("xy"))),)
    if isinstance(v, (Rows, list, tuple, set, frozenset)) and not v:
        raise CodeError(f"{who}() got an empty list -- there is "
                        "nothing there to measure.")
    if isinstance(v, Rows):
        return tuple((r.get("fid"), _tuple_xy(r.get("xy")))
                     for r in v._rows())
    if is_xy(v):
        return ((None, (float(v[0]), float(v[1]))),)
    if isinstance(v, (list, tuple, set, frozenset)):
        return tuple(p for x in v for p in _places(x, who))
    if isinstance(v, str):
        if fid_like(v):
            return ((v, None),)
        # A NAME HAS NO GEOMETRY until find retrieves it.
        raise CodeError(
            f"{who}() measures between things on the surface, and {v!r} is "
            f"a name. find({v!r}) first, then measure against the rows it "
            "serves -- a set is one place and means its nearest member.")
    raise CodeError(
        f"{who}() takes a row, a result set, a fid or an [x, y] point; got "
        f"{type(v).__name__} ({v!r}).")
