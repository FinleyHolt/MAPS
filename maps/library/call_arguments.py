"""A library call's arguments and its ack: the loop's verbs a call runs on
(:class:`MissionEnv`), the model's names for its sets, the argument slots
and place tokens the compiler reads, the grounding that opens a call and the
ack that closes it."""

from __future__ import annotations

import ast
import inspect
import sys
from dataclasses import dataclass
from typing import Callable

from maps.library.op_compiler import (
    CARDINAL_ORDER, HANDLE_RE, VIEWPOINT_RE, Bind, CodeError, Program, Ref,
    act_needs_a_thing, fid_like, rn_spelled)
from maps.library.rows import Row, Rows, UNNAMED, _index_it, _one_row, _set_as_list
from maps.withhold import withheld
from maps.xy_token import format_xy_token, is_xy, parse_xy_token


# Each act's place slot as the model spells it, by the act's ``why``.
_SITE_SLOT = {"map": "around=", "boost": "at=", "stage": "at=", "facing": "facing=",
              "crops": "each look() place", "marks": "each mark",
              "move": "go_to()", "cover": "each explore() place",
              "anchor": "each place a relation names"}


class _Stop(Exception):
    """A plan-level refusal already folded into the receipt: end the block."""


# A block can print type(e) or type(rows).mro(), so the module this class reports is
# served text. It is pinned to the string every recorded run was served.
_Stop.__module__ = "maps.search.parse_compile.python_surface"


def _no_such_fid(fid: str) -> CodeError:
    """The refusal of a prior_/detected_ fid no surface row carries."""
    return CodeError(f"there is no {fid} on this surface; say a fid off a "
                     "served row.")


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


class _CallArguments:
    """What every library call shares: the model's names for its sets, a
    Python value as the argument the compiler reads, a place as its token,
    the grounding that opens a call and the ack that closes it."""

    def name_of(self, obj) -> str | None:
        for k, v in self.ns.items():
            if v is obj and not k.startswith("__"):
                return k
        return getattr(obj, "_label", "") or None

    def spelling(self, x: "Rows") -> str | None:
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
            if isinstance(v, Rows) and not v._spec and v._ref == ref \
                    and not k.startswith("__"):
                return k
        return self._shown.get(ref) or self._labels.get(ref)

    def said_name(self, x: "Rows", fn: str | None) -> str:
        """What a refusal or a page calls ``x``: its bound name, else the
        model's text for it in the running ``fn(...)`` call (:meth:`call_arg`),
        else its spelling, else unnamed."""
        return (self.name_of(x) or self.call_arg(fn) or self.spelling(x)
                or UNNAMED)

    def call_arg(self, fn: str | None) -> str | None:
        """The model's text for the first argument of the ``fn(...)`` call
        its block is running (``fn`` None: for what the subscript it is
        running indexes)."""
        node = self.running_node()
        if fn is None and isinstance(node, ast.Subscript):
            return ast.unparse(node.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == fn and node.args):
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
            for n in ast.walk(ast.parse(self._code)):
                if isinstance(n, (ast.Call, ast.Subscript)):
                    self._nodes.setdefault((n.lineno, n.end_lineno, n.col_offset,
                                            n.end_col_offset), n)
        return self._nodes.get(tuple(inspect.getframeinfo(fr, context=0).positions))

    def held_refs(self) -> set:
        """The executor refs the namespace's sets hold (a window holds its
        whole ref)."""
        return {v._ref for v in self.ns.values() if isinstance(v, Rows)}

    def _held(self, name: str):
        """What a bound name holds, as an argument: a whole result, a row,
        or a staged viewpoint; None for anything else."""
        v = dict.get(self.ns, name)
        if isinstance(v, Rows) and not v._spec:
            return Ref(v._ref, name)
        if isinstance(v, Row):
            return Bind(v._bind, name)
        if isinstance(v, str) and v == name and VIEWPOINT_RE.fullmatch(v):
            return Bind(v, name)
        return None

    def _slot(self, v):
        """A Python value -> the argument the compiler reads: a result is a
        :class:`Ref` and a row a :class:`Bind`, labelled with the model's
        own name (as is a string naming one); anything else rides verbatim,
        a bare name to ground."""
        if isinstance(v, Rows):
            # A nameless set is numbered, once per mission, so two of them
            # in one call stay two places (two columns, two cuts).
            name = self.spelling(v)
            if v._spec:
                v = self._materialize(v)
            return Ref(v._ref, name or self._unnamed_label(v._ref))
        if isinstance(v, Row):
            return Bind(v._bind, self.name_of(v) or (v.fid or "row"))
        if isinstance(v, str):
            return self._held(v) or v
        if isinstance(v, dict):
            raise CodeError("a place is a row, a set, a fid or an [x, y]; "
                            f"got a dict ({v!r:.60}).")
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
            k for k, v in self.ns.items() if HANDLE_RE.fullmatch(k)
            and (isinstance(v, Row) or isinstance(v, Rows) and not v._spec)}
        live = {v._ref for v in self.ns.values()
                if isinstance(v, Rows) and not v._spec}
        return Program(self.env.plan_rn[0], taken, live)

    def _end(self, ack: dict) -> None:
        """Close one call: finish an ack nothing refused, fold it into the
        receipt, bind what its flights minted; a refusal ends the block. A
        call may have changed the surface, so measured geometry is dropped."""
        if "error" not in ack:
            ack = self.env.finish(ack)
        self._geom.clear()
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
                if not n.strip():
                    raise CodeError(
                        f"{_SITE_SLOT[w]} got an empty name, which is no place.")
                why.setdefault(n, w)
        got = self.env.ground(why)
        if "error" in got:
            self._end(got)
        return got["ties"]

    def _site(self, v, why: str):
        """An act's PLACE -> its token, or a :class:`_Name` to ground. A
        point rides as its ``[x, y]`` token; what a binding holds and a fid
        the model typed are never looked up."""
        if v is None and why != "marks":
            raise CodeError(
                f"{_SITE_SLOT[why]} got None, which is no place: the name it "
                "came from holds no row. Bind a row first.")
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
                raise CodeError(f"{_SITE_SLOT[why]} takes ONE place and that "
                                "list holds 0 row(s).")
            raise CodeError(
                f"{_SITE_SLOT[why]} takes ONE place and that is a list of "
                f"{len(v)} -- index the one you mean, or give one [x, y] point.")
        # A QUOTED [x, y] IS A POINT: welcome in a relating slot, refused as
        # a COORDINATE in a moving one.
        pt = parse_xy_token(v) if isinstance(v, str) else None
        if pt:
            if why in ("move", "cover") and not withheld("prior"):
                raise CodeError(act_needs_a_thing(
                    name="go_to/explore", got=f"{pt[0]:g}, {pt[1]:g}")
                    + ("" if withheld("library") else
                       " A crossing you computed is a ROW: index it "
                       "-- r2 = intersection(\"A Road\", \"B St\") then "
                       "go_to(r2[0])."))
            return format_xy_token(float(pt[0]), float(pt[1]))
        if isinstance(v, str) and not v.strip() and why == "marks":
            return v                # a blank mark marks nothing, as None does
        if isinstance(v, str) and not fid_like(v):
            return _Name(v)
        return str(_plain(v))

    def _cards(self, **anchors) -> list:
        """The cardinal cuts given, slotted in the order named, read in
        the compiler's order."""
        got = {d: self._slot(a) for d, a in anchors.items() if a is not None}
        return [(d, got[d]) for d in CARDINAL_ORDER if d in got]

    def _act(self, verb, sites: list, *args, **kw) -> None:
        """Ground the NAMES among ``sites`` (``(site, why)`` pairs), then run
        the act ``verb`` on their tokens."""
        ties = self._ground(sites)
        self._end(verb(*[_tok(s, ties) for s, _w in sites], *args, **kw))

    def _show_as(self, ref: str, label: str | None) -> None:
        """Serve ``ref``'s envelope under ``label``, kept whole (unnamed when
        ``label`` is None)."""
        if label is not None:
            self._shown[ref] = label
        self._block_shown.add(ref)

    def _analyze(self, p: Program, ref: str, *, shown=None, page=None) -> str:
        """Ground the program's named anchors and run it, ``ref``'s envelope
        the answer (served under ``shown`` when given)."""
        if shown is not None:
            self._show_as(ref, shown)
        ties = self._ground((_Name(n), "anchor") for n in p.anchor_names())
        p.ground(ties)
        ack = self.env.analyze(p.ops, p.rn, ref, page)
        if "results" not in ack.get("result", {"results": None}):
            # deduped: its note says so, while the surface stands
            self._unrendered[ref] = self.env.version()
        self._end(ack)
        return ref

    def _fold(self, part: dict) -> None:
        out = self._out
        for k, v in part.items():
            if k == "result" and isinstance(out.get("result"), dict) \
                    and isinstance(v, dict):
                res = {**(out["result"].get("results") or {}),
                       **(v.get("results") or {})}
                out["result"] = {**out["result"], **v, "results": res}
                if not res:
                    del out["result"]["results"]
            elif k == "note" and out.get("note"):
                out["note"] = f"{out['note']}\n{v}"
            elif k == "flown" and k in out:
                # A second call of one kind keeps the first's part: flights
                # run on in one list, any other kind lists each call's part.
                out[k] = {f: out[k].get(f, []) + v.get(f, [])
                          for f in {**out[k], **v}}
            elif isinstance(v, dict) and k in out:
                prev = out[k]
                out[k] = [*prev, v] if isinstance(prev, list) else [prev, v]
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
        return Rows(self._window(sliced._ref, sliced._spec,
                                 self.spelling(sliced), sliced._ref), self)

    def _unnamed_label(self, ref: str) -> str:
        """A nameless set's label, numbered once per mission."""
        return "(unnamed {})".format(
            self._unnamed.setdefault(ref, len(self._unnamed) + 1))

    def _not_one_place(self, kw: str, x: "Rows") -> str:
        """The refusal of a set where ``kw=`` takes one place."""
        name = self.spelling(x)
        one = _one_row(x, name or "s")
        if one is None:
            return (f"{kw}= takes ONE place and {name or 'that set'} holds "
                    "0 row(s) here")
        return (f"{kw}= takes ONE place and {name or 'that'} is a set. "
                f"{_index_it(name)} -- {kw}={one}")
