"""The compiler behind the mission `run` block: each library call
``python_surface`` executes compiles onto geom_program ops the executor
already runs. A :class:`Program` holds one call's ops; the loop grounds its
named anchors and runs it."""
from __future__ import annotations

import re
from dataclasses import dataclass

from maps.search.analyze_scene.geom_compute import CARDINALS as _GEOM_CARDINALS
from maps.search.analyze_scene.geom_compute import COMPASS_UNIT
from maps.search.analyze_scene.geom_program import sanitize_token
from maps.util.xy_token import format_xy_token, is_xy, parse_xy_token
from maps.search.withhold import withheld

# The four map-fixed directions, in the order their kwargs are read. Taken
# from the executor's own table so the surface can never offer a direction
# the geometry does not compute.
CARDINAL_ORDER = tuple(sorted(_GEOM_CARDINALS))


def split_diagonal(word: str):
    """A diagonal wind -> its two cardinals, or None. Derived, not listed."""
    for a in ("north", "south"):
        if word.startswith(a) and word[len(a):] in ("east", "west"):
            return a, word[len(a):]
    return None


def _winds_holding(cardinal_word: str) -> list:
    """Every served wind that COUNTS AS facing this cardinal: ``south`` is
    satisfied by south, southeast and southwest, since a house sits
    DIAGONALLY from the crossing it is on the corner of. Chaining the two
    cardinals of a diagonal takes the precision back."""
    return sorted(w for w in COMPASS_UNIT
                  if cardinal_word in (split_diagonal(w) or (w,)))


# THE MOVING CALLS TAKE A THING, NEVER A COORDINATE (outside the prior
# withhold): a point is an estimate read off a picture, and the rows carry
# measured positions. One refusal for every spelling of a point.
_ACT_NEEDS_A_THING = (
    "{name}() takes a THING on the map, not a coordinate ({got}).{points}")
_ACT_REACH_A_PART = (
    " To reach a part of something, look_for the class you "
    "want, explore the thing so the segmenter puts rows on it, then "
    "{name}() the row you mean.")


def act_needs_a_thing(*, name: str, got: str) -> str:
    """The refusal of a coordinate where a move takes a thing; its remedy
    names look_for and explore, so a ``look_for`` or ``library`` withhold
    serves the refusal alone, a ``look`` withhold says points are for
    relating only, and a ``map`` or ``library`` withhold names none of its
    calls."""
    lib = withheld("library")
    text = _ACT_NEEDS_A_THING + ("" if withheld("look_for") or lib else _ACT_REACH_A_PART)
    uses = ("RELATING" if withheld("look") else "RELATING and LOOKING") if not lib \
        else "LOOKING"
    slots = ", ".join(s for s, w in (("near=, within=", "library"), ("show_map(around=)", "map"),
                                     ("stand(at=)", "library")) if not withheld(w))
    return text.format(name="go_to" if lib else name, got=got,
                       points=f" Points are for {uses} -- {slots}." if slots else "")


class CodeError(ValueError):
    """A refusal the model can act on: what was said, and what is allowed."""


def refuse_metres_in_within(v):
    """``within=`` takes a PLACE; a bare number is ``dwithin=`` (Shapely's
    names: ``a.within(b)`` is containment, ``dwithin(a, b, d)`` a
    distance)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return
    raise CodeError(
        f"within= keeps what TOUCHES a place and {v!r} is a distance, not a "
        f"place. A gap in metres is dwithin=: "
        f"r1.filter(near=<place>, dwithin={v!r}) keeps what is that close, "
        f"and r1.filter(within=<place>) keeps what touches it.")


def bind_row(row: dict):
    """What a selected row binds to: its fid, except a COMPUTED row binds
    its point -- a crossing's fid IS its ``[x, y]`` token, and a terminus
    (``end_N``) is nowhere on the surface, so it binds its own ``xy``."""
    fid = row.get("fid")
    if isinstance(fid, str) and END_RE.fullmatch(fid):
        xy = row.get("xy")
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            return [float(xy[0]), float(xy[1])]
    pt = parse_xy_token(fid) if isinstance(fid, str) else None
    return list(pt) if pt else fid


def row_token(row: dict) -> str:
    """A selected row as the token a move or a crop takes: what it binds
    (:func:`bind_row`), a point as its ``[x, y]`` token."""
    b = bind_row(row)
    return format_xy_token(*b) if isinstance(b, list) else str(b)


def _is_place_list(v) -> bool:
    """``near=[A, B]`` is TWO PLACES; ``near=[4630, 2965]`` is one point."""
    return (isinstance(v, (list, tuple)) and len(v) >= 2
            and not is_xy(v))


# The token grammar, each matched whole: the fid families the world mints
# (everything else in a place position is a NAME to ground), a staged
# viewpoint, a coverage stop, a terminus, and a result handle with or without
# its window.
_FID_N = "[0-9]+"
FID_RE = re.compile(rf"(?:prior_|detected_|stop_|end_|v){_FID_N}",
                    re.IGNORECASE)
VIEWPOINT_RE = re.compile(rf"v{_FID_N}")
STOP_RE = re.compile(rf"stop_{_FID_N}")
END_RE = re.compile(rf"end_{_FID_N}")
HANDLE_RE = re.compile(r"r[0-9]+")
HANDLE_WORD_RE = re.compile(rf"\b{HANDLE_RE.pattern}\b")


def fid_like(tok) -> bool:
    """Is this string a FID rather than a name?"""
    return isinstance(tok, str) and bool(FID_RE.fullmatch(tok.strip()))


@dataclass(frozen=True)
class Ref:
    """A whole result in an argument: its executor ref, and the label its
    columns carry (the model's own name for it)."""
    ref: str
    label: str


@dataclass(frozen=True)
class Bind:
    """One row in an argument: what it binds to (a fid, or the [x, y] of a
    computed row) and the label its columns carry."""
    bind: object
    label: str


def rn_spelled(vals) -> set:
    """Every ``rN`` a call's arguments spell, labels included: the names the
    program it compiles must not mint."""
    out, todo = set(), list(vals)
    while todo:
        v = todo.pop()
        if isinstance(v, (Ref, Bind)):
            v = v.label
        if isinstance(v, str):
            if HANDLE_RE.fullmatch(v):
                out.add(v)
        elif isinstance(v, dict):
            todo.extend(v.values())
        elif isinstance(v, (list, tuple)):
            todo.extend(v)
    return out


def _ref_label(r) -> str:
    """The label an executor ref's columns carry, through the executor's OWN
    sanitizer."""
    return sanitize_token(r["as"] if isinstance(r, dict) and "as" in r else r)


class Program:
    """One call's ops over the mission-global rN chain. The executor's refs and
    the model's handles are both rN, so a program never mints an rN a name in
    play (``taken``) spells; ``live`` holds the executor refs the mission's
    handles hold, and any other bare name is grounded."""

    def __init__(self, rn: int, taken: set, live: set):
        self.ops: list[dict] = []
        self.rn = rn
        self.taken = taken
        self.live = set(live)

    def add(self, op: str, args: dict) -> str:
        while f"r{self.rn}" in self.taken:
            self.rn += 1
        ref = f"r{self.rn}"
        self.rn += 1
        self.ops.append({"ref": ref, "op": op, "args": args})
        return ref

    def ref_of(self, v):
        """An argument -> an executor ref, LABELLED (``as``) with the code's
        own name, so the compare's column and the key that names it agree."""
        if isinstance(v, Ref):
            # A WHOLE RESULT IN A PLACE SLOT IS ITS UNION: one column, the
            # gap to its NEAREST member ("the house with a pool" is about
            # whichever pool is nearest each house).
            self.live.add(v.ref)
            return {"ref": [v.ref], "as": v.label}
        if isinstance(v, Bind):
            # A ROW -- a point or a fid. The wrapper stops a bound fid being
            # read as a bare name and grounded again.
            b = v.bind
            return {"ref": list(b) if isinstance(b, (list, tuple)) else b,
                    "as": v.label}
        if is_xy(v):
            # an inline point takes the same wrapper
            return {"ref": [float(v[0]), float(v[1])],
                    "as": f"{v[0]:g} {v[1]:g}"}
        if isinstance(v, (list, tuple)):
            # AN ANCHOR MAY BE A SET OF OBJECTS. A set folds to its UNION --
            # ONE ref, ONE column, ONE cut; two cuts against two anchors is
            # `.filter(east_of=A).filter(east_of=B)`.
            if not v:
                raise CodeError(
                    "an empty list names no place. Give the anchor: one "
                    "fid, one handle, an [x, y], or a list of them.")
            if len(v) == 1:
                return self.ref_of(v[0])     # [one thing] is the thing
            members, labels = [], []
            for m in v:
                r = self.ref_of(m)
                inner = r["ref"] if isinstance(r, dict) and "ref" in r else r
                if isinstance(inner, list) and len(inner) == 1 \
                        and isinstance(inner[0], str):
                    inner = inner[0]         # a whole result: its rN
                if not isinstance(inner, str) and not (
                        isinstance(m, Bind) and is_xy(inner)):
                    # A union member rides its MASK, or a row with no fid
                    # its own point; a bare [x, y] names no thing.
                    raise CodeError(
                        f"a set anchor holds PLACES -- fids, handles, or "
                        f"results -- and {m!r} is a bare point. Name one "
                        f"point on its own (east_of=[x, y]), or name the "
                        f"things: east_of=[\"prior_12\", \"prior_13\"].")
                members.append(inner)
                labels.append(r["as"] if isinstance(r, dict) and "as" in r
                              else inner)
            label = "_".join(str(x) for x in labels[:2])
            if len(labels) > 2:
                label = f"{label}_plus{len(labels) - 2}"
            return {"ref": members, "as": label}
        # a quoted [x, y] token (a crossing's fid) is a point, not a name
        pt = parse_xy_token(v) if isinstance(v, str) else None
        if pt:
            return {"ref": [float(pt[0]), float(pt[1])],
                    "as": f"{pt[0]:g} {pt[1]:g}"}
        return v

    def items_of(self, v):
        """The `items` slot: the same lookup without the label -- a bare ref
        or a list."""
        got = self.ref_of(v)
        if isinstance(got, dict) and "ref" in got:
            got = got["ref"]
        if (isinstance(got, list) and len(got) == 1
                and isinstance(got[0], str)
                and HANDLE_RE.fullmatch(got[0])):
            return got[0]
        return got

    def label(self, anchor) -> str:
        """The label a compare's columns carry for this anchor, through the
        executor's OWN sanitizer ("the depot" -> "the_depot"), so a key
        built here never names a column that does not exist."""
        return _ref_label(self.ref_of(anchor))

    def cut_le(self, items, anchor, value) -> str:
        """Keep the rows within ``value`` metres of ``anchor``: one compare
        and one ``le`` cut on the distance column it writes."""
        cmp_ref = self.add("compare", {"items": items,
                                       "refs": [self.ref_of(anchor)]})
        return self.add("filter_items", {
            "items": cmp_ref, "key": f"distance to {self.label(anchor)}",
            "op": "le", "value": value})

    def window(self, ref: str, spec: str = "") -> str:
        """The IDENTITY WINDOW: ``filter_items`` truthy on ``xy`` drops
        nothing (every served row carries an xy), so it makes an existing
        result -- or a slice of one -- the answer of a program."""
        return self.add("filter_items", {"items": f"{ref}{spec}", "key": "xy",
                                         "op": "truthy"})

    def relate(self, items, near=(), cards=(), color=None,
               rank: bool = True) -> str:
        """One compare carrying every anchor, a cut per cardinal and, when
        ranking, one score_by for every ordering (a second sort would undo the
        first). A cardinal cuts and proximity to the same anchor orders; the
        compare dedups its refs by label, and a colour orders in the same
        emphasis as the distances."""
        refs, emphasis, seen = [], [], set()
        for a in [*near, *(a for _d, a in cards)]:
            r = self.ref_of(a)
            lab = _ref_label(r)
            if lab in seen:
                continue
            seen.add(lab)
            refs.append(r)
            emphasis.append({"key": f"distance to {lab}", "w": -1})
        args = {"items": items, "refs": refs}
        if cards:
            args["cardinals"] = [{"dir": d, "ref": self.label(a)}
                                 for d, a in cards]
        ref = self.add("compare", args)
        # `on_side` keeps the rows the column could not place -- unmeasured,
        # not refuted -- and conjoined constraints chain.
        for d, a in cards:
            ref = self.add("filter_items", {
                "items": ref, "key": f"{d} of {self.label(a)}",
                "op": "on_side"})
        if not rank:
            return ref
        return self.rank_color(ref, color, emphasis)

    def rank_color(self, items, color, emphasis=()) -> str:
        """One ``score_by`` over ``emphasis`` with the colour, when given,
        ranked last."""
        emphasis = list(emphasis)
        if color:
            emphasis.append({"key": f"color_{color}", "w": 1})
        return self.add("score_by", {"items": items, "emphasis": emphasis})

    def anchor_names(self) -> list:
        """The bare NAMES in the ops' refs, in order: places to ground."""
        live = self.live | {op["ref"] for op in self.ops}
        return [v for op in self.ops for v in op["args"].get("refs") or []
                if isinstance(v, str) and v not in live and not fid_like(v)]

    def ground(self, ties: dict) -> None:
        """Each named anchor -> its grounded fid, or its top-tie fids as the
        labeled UNION ref: one label, the exact min gap over members, never
        a per-instance fan-out."""
        live = self.live | {op["ref"] for op in self.ops}
        for op in self.ops:
            refs = op["args"].get("refs") or []
            for i, v in enumerate(refs):
                if isinstance(v, str) and v not in live and not fid_like(v):
                    fids = ties[v]
                    refs[i] = {"ref": fids[0] if len(fids) == 1 else fids,
                               "as": v}


def _anchors(v) -> list:
    """``nearest=`` -> its anchors: a LIST of places is a set to fuse."""
    return list(v) if _is_place_list(v) else [v]


def _metres(dwithin, example: str) -> float:
    try:
        return float(dwithin)
    except (TypeError, ValueError):
        raise CodeError(
            f"dwithin= is a distance in metres and {dwithin!r} is not a "
            f"number: {example}") from None


def compile_find(p: Program, query, *, name=None, near=None, nearest=None,
                 dwithin=None, within=None, cards=(), color=None) -> str:
    ref = p.add("find", {"query": query} if name is None
                else {"query": query, "name": name})
    # ONE WORD, ONE OPERATION: near= reads like a filter, so it CUTS and
    # needs the gap to cut at; the ordering is nearest=, the word .order()
    # uses.
    if near is not None and dwithin is None:
        raise CodeError(
            "near= CUTS and needs the gap to cut at: "
            "find(<thing>, near=<place>, dwithin=25) keeps what is "
            "that close and DROPS the rest. To rank without "
            "dropping, say nearest=<place>.")
    if dwithin is not None:
        # dwithin= CUTS AGAINST WHICHEVER ANCHOR WAS NAMED, near= or
        # nearest= -- there is one anchor either way.
        anchor = near if near is not None else nearest
        if anchor is None:
            raise CodeError(
                "dwithin= is the gap to cut at and needs the place "
                "to measure from: find(<thing>, near=<place>, "
                f"dwithin={dwithin!r}), or nearest=<place> to rank "
                "against the same place while you cut.")
        # A LIST in near= is one place, its union; in nearest= it ranks by
        # each member's distance.
        if anchor is nearest and _is_place_list(nearest):
            raise CodeError(
                "dwithin= cuts against ONE place, and a LIST in "
                "nearest= ranks by several distances, so there is no "
                "single gap to cut. Say near=<the set> to cut against "
                "its nearest member, or cut once per anchor.")
        wv = _metres(dwithin, "find(<thing>, near=<place>, dwithin=25). A "
                     "PLACE to touch is within=.")
        if wv < 0:
            raise CodeError(
                f"dwithin= is a gap in metres and cannot be negative "
                f"({dwithin!r}).")
        ref = p.cut_le(ref, anchor, wv)
    if within is not None:
        # within= CUTS: 0 m contact is a measured geometric fact, and it
        # stays its own compare + filter ahead of every ordering.
        ref = p.cut_le(ref, within, 0)
    if nearest is not None or cards:
        return p.relate(ref, [] if nearest is None else _anchors(nearest),
                        cards, color)
    if color:
        return p.rank_color(ref, color)
    return ref


def compile_order(p: Program, items, *, nearest=None, color=None) -> str:
    items = p.items_of(items)
    if nearest is not None:
        return p.relate(items, _anchors(nearest), (), color)
    if not color:
        raise CodeError("order ranks by nearest= or color=")
    return p.rank_color(items, color)


def compile_filter(p: Program, items, *, within=None, near=None,
                   dwithin=None, facing=None, direction=None,
                   cards=()) -> str:
    items = p.items_of(items)
    if dwithin is not None and near is None:
        raise CodeError(
            "dwithin= is the gap of a near= cut: r1.filter(near=<place>, "
            "dwithin=25); it rides no other relation.")
    # WHICH WAY A THING FACES A PLACE is a CUT on the `facing <ref>` compass
    # word every compare stamps, and it needs the anchor. The word rides as
    # DATA in direction=; four cardinals, a diagonal is two of them CHAINED.
    if facing is not None or direction is not None:
        if facing is None:
            raise CodeError(
                "direction= says WHICH WAY, and a cut needs what "
                "it faces: rows.filter(facing=<place>, "
                'direction="south"). To RANK by a direction '
                "instead, that is a global question and takes no "
                'anchor: rows.order(direction="north").')
        if direction is None:
            raise CodeError(
                "facing= needs the way it faces: "
                'rows.filter(facing=<place>, direction="south") '
                "keeps the rows whose 'facing <place>' column "
                f"reads south. One of {'|'.join(CARDINAL_ORDER)}.")
        w = str(direction).strip().lower()
        if w not in CARDINAL_ORDER:
            pair = split_diagonal(w)
            if pair:
                raise CodeError(
                    f"a thing facing {w} faces {pair[0]} AND "
                    f"{pair[1]}, so it is two cuts and they "
                    f"chain: rows.filter(facing=<place>, "
                    f'direction="{pair[0]}").filter('
                    f'facing=<place>, direction="{pair[1]}").')
            raise CodeError(
                f"{direction!r} is not a compass direction; "
                f"direction= takes one of {'|'.join(CARDINAL_ORDER)} "
                f"(a diagonal is those two, chained).")
        if cards or within is not None or near is not None:
            raise CodeError(
                "filter() cuts on ONE relation at a time: which "
                "way a row faces is its own cut. Chain them -- "
                'rows.filter(near=<place>, dwithin=25).filter('
                f'facing=<place>, direction="{w}") -- so each '
                "one says what it dropped.")
        cmp_ref = p.add("compare", {"items": items,
                                    "refs": [p.ref_of(facing)]})
        # in_or_blank over the winds that HOLD this cardinal, and the blank
        # stays kept: `facing <ref>` is ABSENT at contact, and a house
        # touching the road it fronts is the common case.
        return p.add("filter_items", {
            "items": cmp_ref, "key": f"facing {p.label(facing)}",
            "op": "in_or_blank", "value": _winds_holding(w)})
    if cards and (within is not None or near is not None):
        raise CodeError(
            "filter() cuts on ONE relation at a time: a cardinal is "
            "its own cut. Chain them -- "
            "rows.filter(near=<place>, dwithin=25).filter"
            f"({cards[0][0]}_of=<place>) -- so each one says what "
            "it dropped.")
    if cards:
        # A CARDINAL IS A CUT: a row the geometry cannot place keeps its
        # blank column and is KEPT.
        return p.relate(items, (), cards, rank=False)
    if within is not None and near is not None:
        raise CodeError(
            "filter() takes within= or near=, not both: within= "
            "keeps what TOUCHES a place, near= keeps what is "
            "within a stated gap of one. Say the one you mean.")
    if within is None and near is None:
        raise CodeError(
            "filter needs what to cut against: r1.filter(within=r0) "
            "keeps what touches r0, r1.filter(near=r0, dwithin=25) "
            "keeps what is within 25 m of it.")
    if near is not None and dwithin is None:
        raise CodeError(
            "near= needs the gap to keep: r1.filter(near=<place>, "
            "dwithin=<meters>). Without one there is no cut; to rank "
            "by nearness, rows.order(nearest=<place>).")
    # within= is the `distance to <anchor>` cut at 0 (contact); a stated gap
    # is the same cut at its value.
    value = 0
    if dwithin is not None:
        value = _metres(dwithin, "r1.filter(near=<place>, dwithin=25).")
        if value < 0:
            raise CodeError(
                f"dwithin= is a gap in metres and cannot be negative "
                f"({dwithin!r}). dwithin=0 is contact, which is "
                "within=.")
    return p.cut_le(items, within if within is not None else near, value)
