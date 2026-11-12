"""What the library's calls are built from: the loop's verbs a call runs on
(:class:`MissionEnv`), the model's names for its sets (:class:`Names`), a
place as the token an act takes, and the fold of a call's ack into the
block's receipt."""

from __future__ import annotations

import ast
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass

from maps.library.op_compiler import Bind, CodeError, Ref, act_needs_a_thing, fid_like
from maps.library.rows import UNNAMED, Rows, _index_it, _one_row
from maps.xy_token import format_xy_token, is_xy, parse_xy_token

# Each act's place slot as the model spells it, by the act's ``why``.
_SITE_SLOT = {
    "map": "around=",
    "boost": "at=",
    "stage": "at=",
    "facing": "facing=",
    "crops": "each look() place",
    "marks": "each mark",
    "move": "go_to()",
    "cover": "each explore() place",
    "anchor": "each place a relation names",
}


class _Stop(Exception):
    """A plan-level refusal already folded into the receipt: end the block."""


def _no_such_fid(fid: str) -> CodeError:
    """The refusal of a prior_/detected_ fid no surface row carries."""
    return CodeError(f"there is no {fid} on this surface; say a fid off a served row.")


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


@dataclass
class MissionEnv:
    """The loop's verbs, one set per mission. A library call opens with
    ``ground(names)`` (``{name: why}`` -> ``{"ties": {name: fids}}`` or an
    error), runs one verb and, unless refused, closes with ``finish(ack)``; a
    refusing ack part carries ``"error"``. ``plan_rn`` is the shared ref
    counter, ``widened`` the paging ledger, ``minted`` the handles the call's
    flights minted, ``say_err(ack, msg)`` the refusal format,
    ``row_of_fid(fid)`` a live-row reader (None when absent),
    ``tied_of_ref(ref)`` whether a ref holds a tied view and ``version()``
    the surface version."""

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
    tied_of_ref: Callable
    version: Callable


def place_token(v, why: str):
    """An act's PLACE -> its token, or a :class:`_Name` to ground. A
    point rides as its ``[x, y]`` token; what a binding holds and a fid
    the model typed are never looked up."""
    if v is None and why != "marks":
        raise CodeError(
            f"{_SITE_SLOT[why]} got None, which is no place: the name it "
            "came from holds no row. Bind a row first."
        )
    if isinstance(v, Ref):
        return v.ref
    if isinstance(v, Bind):
        b = v.bind
        if isinstance(b, (list, tuple)) and len(b) == 2:
            return format_xy_token(float(b[0]), float(b[1]))
        return b if isinstance(b, str) else str(v.label)
    if isinstance(v, (list, tuple)):
        v = _plain(v)
        if is_xy(v):
            return format_xy_token(float(v[0]), float(v[1]))
        if not v:
            raise CodeError(f"{_SITE_SLOT[why]} takes ONE place and that list holds 0 row(s).")
        raise CodeError(
            f"{_SITE_SLOT[why]} takes ONE place and that is a list of "
            f"{len(v)} -- index the one you mean, or give one [x, y] point."
        )
    # A QUOTED [x, y] IS A POINT: welcome in a relating slot, refused as
    # a COORDINATE in a moving one.
    pt = parse_xy_token(v) if isinstance(v, str) else None
    if pt:
        if why in ("move", "cover"):
            raise CodeError(
                act_needs_a_thing(name="go_to/explore", got=f"{pt[0]:g}, {pt[1]:g}")
                + " A crossing you computed is a ROW: index it "
                '-- r2 = intersection("A Road", "B St") then '
                "go_to(r2[0])."
            )
        return format_xy_token(float(pt[0]), float(pt[1]))
    if isinstance(v, str) and not v.strip() and why == "marks":
        return v  # a blank mark marks nothing, as None does
    if isinstance(v, str) and not fid_like(v):
        return _Name(v)
    return str(_plain(v))


def fold(out: dict, part: dict) -> None:
    """Fold one call's ack ``part`` into the block's receipt ``out``."""
    for k, v in part.items():
        if k == "result" and isinstance(out.get("result"), dict) and isinstance(v, dict):
            res = {**(out["result"].get("results") or {}), **(v.get("results") or {})}
            out["result"] = {**out["result"], **v, "results": res}
            if not res:
                del out["result"]["results"]
        elif k == "note" and out.get("note"):
            out["note"] = f"{out['note']}\n{v}"
        elif k == "flown" and k in out:
            # A second call of one kind keeps the first's part: flights
            # run on in one list, any other kind lists each call's part.
            out[k] = {f: out[k].get(f, []) + v.get(f, []) for f in {**out[k], **v}}
        elif isinstance(v, dict) and k in out:
            prev = out[k]
            out[k] = [*prev, v] if isinstance(prev, list) else [prev, v]
        else:
            out[k] = v


class Names:
    """The model's names for its sets, read off the namespace it binds them
    in and the block it is running."""

    def __init__(self, namespace: dict):
        self.ns = namespace
        #: Model spelling for every ref it was shown rows under, so the
        #: served envelope is keyed by the model's own name.
        self.shown: dict = {}
        #: ref -> the call that made it (find).
        self.labels: dict = {}
        self._unnamed: dict = {}  # nameless set -> its slot number
        # The running block: its source, its code objects and (parsed once
        # a block) its calls and subscripts by position.
        self.code = ""
        self._block_codes: set = set()
        self._nodes: dict | None = None

    def begin_block(self, code: str) -> None:
        """A block arrives: ``code`` is its source."""
        self.code = code
        self._nodes = None

    def running(self, block_codes: set) -> None:
        """The block compiled: these are its code objects."""
        self._block_codes = block_codes

    def name_of(self, obj) -> str | None:
        for k, v in self.ns.items():
            if v is obj and not k.startswith("__"):
                return k
        return getattr(obj, "_label", "") or None

    def spelling(self, x: Rows) -> str | None:
        """The model's spelling of a set, the one name every served label
        uses: its bound name, else its whole result's name with the window;
        None when the model has no name for it (never the executor's rN)."""
        name = self.name_of(x)
        if name:
            return name
        base = self.ref_name(x._ref)
        return base and f"{base}{x._spec}"

    def ref_name(self, ref: str) -> str | None:
        """The model's name for a whole result: a bound name, else what it
        was shown under or the call that made it; None when it has none."""
        for k, v in self.ns.items():
            if isinstance(v, Rows) and not v._spec and v._ref == ref and not k.startswith("__"):
                return k
        return self.shown.get(ref) or self.labels.get(ref)

    def said_name(self, x: Rows, fn: str | None) -> str:
        """What a refusal or a page calls ``x``: its bound name, else the
        model's text for it in the running ``fn(...)`` call (:meth:`call_arg`),
        else its spelling, else unnamed."""
        return self.name_of(x) or self.call_arg(fn) or self.spelling(x) or UNNAMED

    def call_arg(self, fn: str | None) -> str | None:
        """The model's text for the first argument of the ``fn(...)`` call
        its block is running (``fn`` None: for what the subscript it is
        running indexes)."""
        node = self.running_node()
        if fn is None and isinstance(node, ast.Subscript):
            return ast.unparse(node.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == fn
            and node.args
        ):
            return ast.unparse(node.args[0])
        return None

    def running_node(self) -> ast.Call | ast.Subscript | None:
        """The call or subscript the block is running, found at the block
        frame's live position; None outside one."""
        fr = sys._getframe()
        while fr is not None and fr.f_code not in self._block_codes:
            fr = fr.f_back
        if fr is None:
            return None
        if self._nodes is None:
            # the model's source, parsed once a block; the compiled tree
            # is instrumented
            self._nodes = {}
            for n in ast.walk(ast.parse(self.code)):
                if isinstance(n, (ast.Call, ast.Subscript)):
                    self._nodes.setdefault(
                        (n.lineno, n.end_lineno, n.col_offset, n.end_col_offset), n
                    )
        return self._nodes.get(tuple(inspect.getframeinfo(fr, context=0).positions))

    def held_refs(self) -> set:
        """The executor refs the namespace's sets hold (a window holds its
        whole ref)."""
        return {v._ref for v in self.ns.values() if isinstance(v, Rows)}

    def unnamed_label(self, ref: str) -> str:
        """A nameless set's label, numbered once per mission."""
        return f"(unnamed {self._unnamed.setdefault(ref, len(self._unnamed) + 1)})"

    def not_one_place(self, kw: str, x: Rows) -> str:
        """The refusal of a set where ``kw=`` takes one place."""
        name = self.spelling(x)
        one = _one_row(x, name or "s")
        if one is None:
            return f"{kw}= takes ONE place and {name or 'that set'} holds 0 row(s) here"
        return (
            f"{kw}= takes ONE place and {name or 'that'} is a set. {_index_it(name)} -- {kw}={one}"
        )
