"""The compiler behind the mission `run` block: steps -> geom_program ops.

``python_surface`` executes the model's Python and builds one
``{target, call, args}`` step per mission call; this module turns a list
of those steps into the sentence-shaped plan
``reasoner_loop._run_plan`` executes. Every call lands on an op the
executor already runs -- ``find`` is find, ``intersection`` is crossings,
``ends_of`` is ends -- so the surface adds NO execution model, and a code
plan and a sentence plan reach the same grounding, the same name-match
floor, the same flights and the same receipts.

``compile_steps`` mints the executor refs and wires the op DAG;
``_plan_of`` wraps that program in the plan dict, grounding every bare
name through the same plumbing find a sentence's place goes through
(``_site``/``_facing``). It holds no model call and no I/O.
"""
from __future__ import annotations

import re

from maps.search.analyze_scene.cardinal_side import CARDINALS as _GEOM_CARDINALS
from maps.search.analyze_scene.geom_compute import COMPASS_UNIT
from maps.search.analyze_scene.geom_program import _BROAD_HEAD_ROWS, _sanitize_token
from maps.util.xy_token import format_xy_token, is_xy, parse_xy_token
from maps.search.withhold import withheld

# The four map-fixed directions, in the order their kwargs are read. Taken
# from the executor's own table so the surface can never offer a direction
# the geometry does not compute.
_CARDINALS = tuple(sorted(_GEOM_CARDINALS))


def _split_diagonal(word: str):
    """A diagonal wind -> its two cardinals, or None. Derived, not listed."""
    for a in ("north", "south"):
        if word.startswith(a) and word[len(a):] in ("east", "west"):
            return a, word[len(a):]
    return None


def _wind_parts(word: str) -> frozenset:
    """The CARDINALS a compass wind is made of.

    A DIRECTION COMPOSES. Southwest is south and west, so a thing facing
    southwest faces south and it faces west. Derived from the wind's own
    spelling rather than tabulated, so the eight winds and the four
    cardinals cannot disagree.
    """
    pair = _split_diagonal(word)
    return frozenset(pair) if pair else frozenset((word,))


def _winds_holding(cardinal_word: str) -> list:
    """Every served wind that COUNTS AS facing this cardinal.

    ``south`` is satisfied by south, southeast and southwest: a house sits
    DIAGONALLY from the crossing it is on the corner of, so the column reads
    `southwest` where the cue says `south` (exact-word matching held 12.1% of
    corner-facing cues on ground truth, composition 87.9%). The cut is looser
    by construction; chaining the two cardinals of a diagonal takes the
    precision back.
    """
    return sorted(w for w in COMPASS_UNIT if cardinal_word in _wind_parts(w))


# THE MOVING CALLS TAKE A THING, NEVER A COORDINATE (outside the prior
# withhold). One refusal for every spelling of a point -- go_to([x, y]), a
# quoted token, a name bound to a literal -- because a point is the escape
# hatch that answers with an estimate read off a picture instead of a
# measurement; the rows are what carry measured positions.
_ACT_NEEDS_A_THING = (
    "{name}() takes a THING on the map, not a coordinate ({got}). Points "
    "are for RELATING and LOOKING -- near=, within=, show_map(around=), "
    "stand(at=).")
_ACT_REACH_A_PART = (
    " To reach a part of something, look_for the class you "
    "want, explore the thing so the segmenter puts rows on it, then "
    "{name}() the row you mean.")


def act_needs_a_thing(*, name: str, got: str) -> str:
    """The refusal of a coordinate where a move takes a thing; its remedy
    names look_for, so a ``look_for`` withhold serves the refusal alone."""
    text = _ACT_NEEDS_A_THING + ("" if withheld("look_for") else _ACT_REACH_A_PART)
    return text.format(name=name, got=got)


class CodeError(ValueError):
    """A refusal the model can act on: what was said, and what is allowed."""


# ---------------------------------------------------------------------------
# The compiler: steps -> geom_program ops
# ---------------------------------------------------------------------------
# Every call lands on an op the executor already runs: `order` is score_by,
# `filter` is filter_items, `intersection` is crossings. `rN[i]` is the one
# exception -- the executor has no take-a-row op, so indexing resolves
# client-side against the rows that handle holds.


def _refuse_metres_in_within(v):
    """``within=`` takes a PLACE; a bare number is ``dwithin=``.

    The two names are Shapely's: ``a.within(b)`` is containment and
    ``dwithin(a, b, d)`` is a distance. The two arguments have different
    TYPES, so ``within=25`` is caught here rather than served as a cut
    against a "place" called 25.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return
    raise CodeError(
        f"within= keeps what TOUCHES a place and {v!r} is a distance, not a "
        f"place. A gap in metres is dwithin=: "
        f"r1.filter(near=<place>, dwithin={v!r}) keeps what is that close, "
        f"and r1.filter(within=<place>) keeps what touches it.")


def _add_color(emphasis: list, args: dict) -> None:
    """Append the colour term to an emphasis, once, for find and order.

    A colour ORDERS and never drops: the executor rejects a colour filter,
    since a hard cut can lose the true target. It rides the SAME emphasis
    as the distance terms, because a second score_by re-ranks what the
    first ordered and the first then contributes nothing.
    """
    if args.get("color"):
        emphasis.append({"key": f"color_{args['color']}", "w": 1})


def compile_steps(steps: list[dict], rn: int, known: dict,
                  seen: dict) -> tuple[dict, dict, int]:
    """``(program, handles, next_rn)``.

    ``handles`` maps each code name to the op ref that holds its rows, so a
    later block referring to `r1` resolves to the same executor result the
    seed already carries.

    ``seen`` is the mission's widening ledger -- executor ref -> how many
    rows of it the agent has already asked past -- read only by
    ``more_of``. It is read at COMPILE time, and the loop writes it only
    after a plan executes, so a refused widening never advances the window
    and the rows it failed to reach stay reachable.
    """
    ops: list[dict] = []
    # Handles bound by EARLIER segments resolve here too, so an ordering key
    # and the ref it names agree.
    handles: dict = dict(known)
    # THE TWO NAMESPACES ARE KEPT DISJOINT. The executor's refs are rN and
    # so are the model's handles, and an `as` label rides IN the op's args,
    # where the executor's cycle detector reads every string as a possible
    # in-program handle. So this program never mints an rN that any name in
    # play -- argument names included -- already spells.
    _mine = {k for k in handles if re.fullmatch(r"r\d+", k)}
    _seen = list(steps)
    while _seen:
        _v = _seen.pop()
        if isinstance(_v, str):
            if re.fullmatch(r"r\d+", _v):
                _mine.add(_v)
        elif isinstance(_v, dict):
            _seen.extend(_v.values())
        elif isinstance(_v, (list, tuple)):
            _seen.extend(_v)

    def mint():
        nonlocal rn
        while f"r{rn}" in _mine:
            rn += 1
        rn += 1
        return f"r{rn - 1}"

    def ref_of(v):
        """A code value -> an executor ref, LABELLED with the code's own name.

        The column label rule is the executor's: a bare [x, y] labels its
        columns 'distance to xy', so an anchor that is a selected crossing
        would produce a key the ordering cannot name. `{ref, as}` names it
        after the handle the code used, so `r2.order(nearest=r1)` reads
        'distance to r1' and the two halves agree by construction.
        """
        if isinstance(v, str) and v in handles:
            got = handles[v]
            if isinstance(got, str) and re.fullmatch(r"r\d+", got):
                # A WHOLE RESULT IN A PLACE SLOT IS ITS UNION: one column,
                # the gap to its NEAREST member ("the house with a pool" is
                # about whichever pool is nearest each house). `items_of`
                # unwraps this back to the bare ref.
                return {"ref": [got], "as": v}
            # A ROW -- a point or a fid. The wrapper stops a bound fid
            # being read as a bare name and grounded again.
            return {"ref": list(got) if isinstance(got, (list, tuple))
                    else got, "as": v}
        if is_xy(v):
            # AN INLINE POINT TAKES THE SAME WRAPPER: the executor labels a
            # bare [x, y] 'distance to xy' while key_for would name
            # 'distance to 4630_2965', so the label must ride with it.
            return {"ref": [float(v[0]), float(v[1])],
                    "as": f"{v[0]:g} {v[1]:g}"}
        if isinstance(v, (list, tuple)):
            # AN ANCHOR MAY BE A SET OF OBJECTS, and every slot comes
            # through here. A set folds to its UNION -- ONE ref, ONE column,
            # ONE cut; two cuts against two anchors is
            # `.filter(east_of=A).filter(east_of=B)`.
            if not v:
                raise CodeError(
                    "an empty list names no place. Give the anchor: one "
                    "fid, one handle, an [x, y], or a list of them.")
            if len(v) == 1:
                return ref_of(v[0])          # [one thing] is the thing
            # a bare [x, y] never reaches here: the point branch above
            # takes it, which is why that branch stays FIRST
            _members, _labels = [], []
            for _m in v:
                _r = ref_of(_m)
                _inner = _r["ref"] if isinstance(_r, dict) and "ref" in _r \
                    else _r
                if isinstance(_inner, list) and len(_inner) == 1 \
                        and isinstance(_inner[0], str):
                    _inner = _inner[0]       # a whole result: its rN
                if not isinstance(_inner, str):
                    # A union member rides its MASK, and a bare [x, y] has
                    # none, so the executor's union takes fids and rN only.
                    # Name the two shapes rather than let the list reach it.
                    raise CodeError(
                        f"a set anchor holds PLACES -- fids, handles, or "
                        f"results -- and {_m!r} is a bare point. Name one "
                        f"point on its own (east_of=[x, y]), or name the "
                        f"things: east_of=[\"prior_12\", \"prior_13\"].")
                _members.append(_inner)
                _labels.append(_r["as"] if isinstance(_r, dict)
                               and "as" in _r else _inner)
            # The label is DERIVED from the members, so key_for and the
            # compare agree by construction: both call this function.
            _as = "_".join(str(x) for x in _labels[:2])
            if len(_labels) > 2:
                _as = f"{_as}_plus{len(_labels) - 2}"
            return {"ref": _members, "as": _as}
        # A CROSSING'S FID *IS* ITS [x, y] TOKEN, and the model quotes it
        # like any fid; left a string it would be grounded as a NAME.
        _pt = parse_xy_token(v) if isinstance(v, str) else None
        if _pt:
            return {"ref": [float(_pt[0]), float(_pt[1])],
                    "as": f"{_pt[0]:g} {_pt[1]:g}"}
        return v

    def items_of(v):
        """The `items` slot: the SAME lookup, without the label.

        `items` takes a WHOLE result and its coercer wants a bare ref or a
        list, so the ``{ref, as}`` wrapper -- and a union's one-element
        ``["rN"]`` -- comes back off.
        """
        got = ref_of(v)
        if isinstance(got, dict) and "ref" in got:
            got = got["ref"]
        if (isinstance(got, list) and len(got) == 1
                and isinstance(got[0], str)
                and re.fullmatch(r"r\d+", got[0])):
            return got[0]
        return got

    def _label_of(anchor):
        """The label a compare's columns carry for this anchor, through the
        executor's OWN sanitizer ("the depot" -> "the_depot"), so a key
        built here never names a column that does not exist. A compare's
        ``cardinals`` element addresses its ref by this label too: the op
        matches on the ref's fid first and its label second."""
        r = ref_of(anchor)
        lbl = r["as"] if isinstance(r, dict) and "as" in r else r
        return _sanitize_token(lbl)

    def key_for(anchor):
        """The distance column `compare` writes for this anchor."""
        return f"distance to {_label_of(anchor)}"

    def cardinal_key(direction, anchor):
        """The signed column `compare` writes for this relation."""
        return f"{direction} of {_label_of(anchor)}"

    def result_of(src, said: str, maker: str) -> str:
        """The executor ref a handle names, refused unless it is a RESULT."""
        got = handles.get(src) if isinstance(src, str) else None
        if not (isinstance(got, str) and re.fullmatch(r"r\d+", got)):
            raise CodeError(
                f"{said} a RESULT, and {src} is "
                + (f"one row ({got})" if got is not None else "not bound")
                + f". Name a handle {maker} left.")
        return got

    def cut_le(items, anchor, value):
        """Keep the rows within ``value`` metres of ``anchor``: one compare
        and one ``le`` cut on the distance column it writes."""
        cmp_ref = mint()
        ops.append({"ref": cmp_ref, "op": "compare",
                    "args": {"items": items, "refs": [ref_of(anchor)]}})
        out = mint()
        ops.append({"ref": out, "op": "filter_items",
                    "args": {"items": cmp_ref, "key": key_for(anchor),
                             "op": "le", "value": value}})
        return out

    def window(src_ref, spec: str = ""):
        """The IDENTITY WINDOW op: the rows a handle already holds, served.

        ``filter_items`` truthy on ``xy`` drops nothing (every served row
        carries an xy), so its whole job is to make an existing result -- or
        a slice of one -- the answer of a program.
        """
        ref = mint()
        ops.append({"ref": ref, "op": "filter_items",
                    "args": {"items": f"{src_ref}{spec}", "key": "xy",
                             "op": "truthy"}})
        return ref

    for st in steps:
        call, args = st["call"], st["args"]
        pos = args.get("_pos") or []
        if call == "stratum":
            # THE MATCH TIER AS A SET: r1.exact / r1.tagged / r1.fuzzy
            # compile to the executor's own stratum op -- a view over a
            # handle a find left, exactly as a window is.
            src = args["of"]
            got = result_of(src, f"{src}.{args['tier']} takes a tier of",
                            "a find")
            ref = mint()
            ops.append({"ref": ref, "op": "stratum",
                        "args": {"items": got, "tier": args["tier"]}})
            st["_ref"] = ref
            st["_label"] = f"{src}.{args['tier']}"
            continue
        if call == "window":
            # THE TOP-N SLICE (r2[0:20]): the identity window over a
            # handle, so a slice rides every slot a set can.
            src = args["win_of"]
            got = result_of(src, f"{src}{args['spec']} windows",
                            "a find or an ordering")
            ref = window(got, args["spec"])
            st["_ref"] = ref
            if st.get("target"):
                handles[st["target"]] = ref
            st["_label"] = f"{src}{args['spec']}"
            continue
        if st.get("act") or call in ("look", "show_map", "look_for", "stand"):
            continue                       # an ACT is a plan step, not an op
        if call == "ends_of":
            # THE TWO ENDS OF A LINE, as rows a move can take ("the end
            # of", "cul-de-sac").
            ref = mint()
            ops.append({"ref": ref, "op": "ends",
                        "args": {"refs": [ref_of(pos[0])]}})
        elif call == "intersection":
            if len(pos) != 2:
                raise CodeError("intersection takes exactly two roads")
            ref = mint()
            ops.append({"ref": ref, "op": "crossings",
                        "args": {"refs": [ref_of(p) for p in pos]}})
        elif call == "find":
            ref = mint()
            _fargs = {"query": pos[0] if pos else ""}
            if args.get("name") is not None:
                _fargs["name"] = args["name"]
            ops.append({"ref": ref, "op": "find", "args": _fargs})
            # ONE WORD, ONE OPERATION: near= reads like a filter, so it
            # CUTS and needs the gap to cut at; the ordering is nearest=,
            # the word .order() uses.
            if args.get("near") is not None and args.get("dwithin") is None:
                raise CodeError(
                    "near= CUTS and needs the gap to cut at: "
                    "find(<thing>, near=<place>, dwithin=25) keeps what is "
                    "that close and DROPS the rest. To rank without "
                    "dropping, say nearest=<place>.")
            # COLOUR STAYS ON find, against the one-word-one-operation
            # tidiness: a second call for it measured 13 colour-ordering
            # heads over 2 of 500 episodes, against 440 over 175 of 300
            # inline.
            _dwithin = args.get("dwithin")
            if _dwithin is not None:
                # dwithin= CUTS AGAINST WHICHEVER ANCHOR WAS NAMED, near= or
                # nearest= -- there is one anchor either way.
                _anchor = args.get("near")
                if _anchor is None:
                    _anchor = args.get("nearest")
                if _anchor is None:
                    raise CodeError(
                        "dwithin= is the gap to cut at and needs the place "
                        "to measure from: find(<thing>, near=<place>, "
                        f"dwithin={_dwithin!r}), or nearest=<place> to rank "
                        "against the same place while you cut.")
                if isinstance(_anchor, list):
                    raise CodeError(
                        "dwithin= cuts against ONE place, and a LIST in "
                        "near= fuses several distances into an ordering, so "
                        "there is no single gap to cut. Pass the set itself "
                        "-- near=find(\"house\") is one place meaning its "
                        "nearest member -- or cut once per anchor.")
                try:
                    _wv = float(_dwithin)
                except (TypeError, ValueError):
                    raise CodeError(
                        f"dwithin= is a distance in metres and {_dwithin!r} "
                        "is not a number: find(<thing>, near=<place>, "
                        "dwithin=25). A PLACE to touch is within=.")
                if _wv < 0:
                    raise CodeError(
                        f"dwithin= is a gap in metres and cannot be negative "
                        f"({_dwithin!r}).")
                ref = cut_le(ref, _anchor, _wv)
            if args.get("within") is not None:
                # within= CUTS: 0 m contact is a measured geometric fact, and
                # it stays its own compare + filter ahead of every ordering.
                ref = cut_le(ref, args["within"], 0)
            # ONE compare carrying every named anchor, a cut per cardinal,
            # and ONE score_by for every ordering constraint: a second sort
            # re-ranks what the first ordered and the first is lost.
            cards = [(d, args[f"{d}_of"]) for d in _CARDINALS
                     if args.get(f"{d}_of") is not None]
            near_anchor = args.get("nearest")
            if near_anchor is not None or cards:
                # REFS ARE DEDUPED BY LABEL. The executor suffixes a
                # repeated ref's label (`jacaranda_avenue_2`), so a key built
                # here would name a column never written; one corner names
                # the same place twice.
                refs: list = []
                seen_labels: set = set()

                def _add_ref(anchor):
                    lab = _label_of(anchor)
                    if lab not in seen_labels:
                        seen_labels.add(lab)
                        refs.append(ref_of(anchor))

                # A CARDINAL CUTS; PROXIMITY ORDERS. 194 of the corpus's
                # 208 cardinal cues mean ADJACENT ("the house east of 87th
                # Street" is just east of it), so the side is a filter and
                # the ordering is nearness to the same anchor.
                emphasis: list = []
                seen_keys: set = set()

                def _add_near(anchor):
                    k = key_for(anchor)
                    if k not in seen_keys:
                        seen_keys.add(k)
                        emphasis.append({"key": k, "w": -1})

                # TWO ANCHORS ORDER BETTER THAN ONE: ranking by closeness to
                # BOTH features a cue names took top-24 from 68.7% to 92.9%
                # on the 281 episodes whose cue names two. score_by min-max
                # normalizes each key, so two distances fuse evenly.
                for _na in (near_anchor if _is_place_list(near_anchor)
                            else [near_anchor] if near_anchor is not None
                            else []):
                    _add_ref(_na)
                    _add_near(_na)
                for d, a in cards:
                    _add_ref(a)
                    _add_near(a)
                cmp_args = {"items": ref, "refs": refs}
                if cards:
                    cmp_args["cardinals"] = [{"dir": d, "ref": _label_of(a)}
                                             for d, a in cards]
                cmp_ref = mint()
                ops.append({"ref": cmp_ref, "op": "compare", "args": cmp_args})
                # A SIDE CONSTRAINT CUTS: a row the geometry PLACES on the
                # other side is not a candidate. `on_side` keeps the rows the
                # column could not place -- unmeasured, not refuted -- and
                # conjoined constraints chain.
                for d, a in cards:
                    cut = mint()
                    ops.append({"ref": cut, "op": "filter_items",
                                "args": {"items": cmp_ref,
                                         "key": cardinal_key(d, a),
                                         "op": "on_side"}})
                    cmp_ref = cut
                _add_color(emphasis, args)
                if emphasis:
                    out = mint()
                    ops.append({"ref": out, "op": "score_by",
                                "args": {"items": cmp_ref,
                                         "emphasis": emphasis}})
                    ref = out
                else:
                    ref = cmp_ref
            elif args.get("color"):
                # COLOUR WITH NO PLACE: the sort alone.
                emphasis: list = []
                _add_color(emphasis, args)
                out = mint()
                ops.append({"ref": out, "op": "score_by",
                            "args": {"items": ref, "emphasis": emphasis}})
                ref = out
        elif call == "order":
            items = items_of(pos[0])
            emphasis = []
            anchor = args.get("nearest")
            if anchor is not None:
                # nearest= takes ONE place or a LIST of them, as find's
                # does: one compare carrying every anchor, one score_by
                # fusing the normalized distances.
                anchors = (list(anchor) if _is_place_list(anchor)
                           else [anchor])
                refs, seen_labels = [], set()
                for _a in anchors:
                    _lab = _label_of(_a)
                    if _lab not in seen_labels:
                        seen_labels.add(_lab)
                        refs.append(ref_of(_a))
                        emphasis.append({"key": key_for(_a), "w": -1})
                cmp_ref = mint()
                ops.append({"ref": cmp_ref, "op": "compare",
                            "args": {"items": items, "refs": refs}})
                items = cmp_ref
            _add_color(emphasis, args)
            if not emphasis:
                raise CodeError("order ranks by nearest= or color=")
            ref = mint()
            ops.append({"ref": ref, "op": "score_by",
                        "args": {"items": items, "emphasis": emphasis}})
        elif call == "more_of":
            src = pos[0]
            got = result_of(src, f"more_of({src}) pages",
                            "a find or an ordering")
            lo = _BROAD_HEAD_ROWS + int(seen.get(got) or 0)
            ref = window(got, f"[{lo}:{lo + _BROAD_HEAD_ROWS}]")
            st["_page"] = (got, lo)
            # The rows served are NOT the handle -- they are a window of it
            # -- so the envelope is labelled with what was asked for rather
            # than with a name that would claim to be the whole set.
            st["_label"] = f"more_of({src})"
        elif call == "filter":
            items = items_of(pos[0])
            within = args.get("within")
            near = args.get("near")
            dwithin = args.get("dwithin")
            # A CARDINAL IS A CUT on the on_side op find compiles: a row the
            # geometry cannot place keeps its blank column and is KEPT.
            _cards = [(d, args[f"{d}_of"]) for d in _CARDINALS
                      if args.get(f"{d}_of") is not None]
            # WHICH WAY A THING FACES A PLACE is a CUT on the `facing <ref>`
            # compass word every compare stamps, and it needs the anchor
            # ("the south-facing house on Viburnum Avenue"). The word rides
            # as DATA in direction=, never in the kwarg name. Four
            # cardinals; a diagonal is the two of them CHAINED, while the
            # served column still reads all eight winds.
            _facing = args.get("facing")
            _direction = args.get("direction")
            if _facing is not None or _direction is not None:
                if _facing is None:
                    raise CodeError(
                        "direction= says WHICH WAY, and a cut needs what "
                        "it faces: rows.filter(facing=<place>, "
                        'direction="south"). To RANK by a direction '
                        "instead, that is a global question and takes no "
                        'anchor: rows.order(direction="north").')
                if _direction is None:
                    raise CodeError(
                        "facing= needs the way it faces: "
                        'rows.filter(facing=<place>, direction="south") '
                        "keeps the rows whose 'facing <place>' column "
                        f"reads south. One of {'|'.join(_CARDINALS)}.")
                _w = str(_direction).strip().lower()
                if _w not in _CARDINALS:
                    _pair = _split_diagonal(_w)
                    if _pair:
                        raise CodeError(
                            f"a thing facing {_w} faces {_pair[0]} AND "
                            f"{_pair[1]}, so it is two cuts and they "
                            f"chain: rows.filter(facing=<place>, "
                            f'direction="{_pair[0]}").filter('
                            f'facing=<place>, direction="{_pair[1]}").')
                    raise CodeError(
                        f"{_direction!r} is not a compass direction; "
                        f"direction= takes one of {'|'.join(_CARDINALS)} "
                        f"(a diagonal is those two, chained).")
                if _cards or within is not None or near is not None:
                    raise CodeError(
                        "filter() cuts on ONE relation at a time: which "
                        "way a row faces is its own cut. Chain them -- "
                        'rows.filter(near=<place>, dwithin=25).filter('
                        f'facing=<place>, direction="{_w}") -- so each '
                        "one says what it dropped.")
                cmp_ref = mint()
                ops.append({"ref": cmp_ref, "op": "compare",
                            "args": {"items": items,
                                     "refs": [ref_of(_facing)]}})
                ref = mint()
                # in_or_blank over the winds that HOLD this cardinal, and
                # the blank stays kept: `facing <ref>` is ABSENT at contact,
                # and a house touching the road it fronts is the common case.
                ops.append({"ref": ref, "op": "filter_items",
                            "args": {"items": cmp_ref,
                                     "key": f"facing {_label_of(_facing)}",
                                     "op": "in_or_blank",
                                     "value": _winds_holding(_w)}})
                handles[st.get("_ref") or ref] = ref
                st["_ref"] = ref
                continue
            if _cards and (within is not None or near is not None):
                raise CodeError(
                    "filter() cuts on ONE relation at a time: a cardinal is "
                    "its own cut. Chain them -- "
                    "rows.filter(near=<place>, dwithin=25).filter"
                    f"({_cards[0][0]}_of=<place>) -- so each one says what "
                    "it dropped.")
            if _cards:
                cmp_ref = mint()
                ops.append({"ref": cmp_ref, "op": "compare",
                            "args": {"items": items,
                                     "refs": [ref_of(a) for _d, a in _cards],
                                     "cardinals": [
                                         {"dir": d, "ref": _label_of(a)}
                                         for d, a in _cards]}})
                for d, a in _cards:
                    cut = mint()
                    ops.append({"ref": cut, "op": "filter_items",
                                "args": {"items": cmp_ref,
                                         "key": cardinal_key(d, a),
                                         "op": "on_side"}})
                    cmp_ref = cut
                ref = cmp_ref
                handles[st.get("_ref") or ref] = ref
                st["_ref"] = ref
                continue
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
            if within is not None and dwithin is not None:
                raise CodeError(
                    "within= is contact and takes no dwithin=: a gap is "
                    "near='s question. Say r1.filter(near=<place>, "
                    f"dwithin={dwithin!r}) to keep what is that close.")
            if near is not None and dwithin is None:
                raise CodeError(
                    "near= needs the gap to keep: r1.filter(near=<place>, "
                    "dwithin=<meters>). Without one it orders and drops "
                    "nothing, which is find(near=<place>).")
            # within= is the `distance to <anchor>` cut at 0 (contact); a
            # stated gap is the same cut at its value.
            anchor = within if within is not None else near
            if dwithin is None:
                value = 0
            else:
                try:
                    value = float(dwithin)
                except (TypeError, ValueError):
                    raise CodeError(
                        f"dwithin= is a distance in metres and {dwithin!r} is "
                        "not a number: r1.filter(near=<place>, dwithin=25).")
                if value < 0:
                    raise CodeError(
                        f"dwithin= is a gap in metres and cannot be negative "
                        f"({dwithin!r}). dwithin=0 is contact, which is "
                        "within=.")
            ref = cut_le(items, anchor, value)
        else:
            raise CodeError(f"there is no {call}()")
        st["_ref"] = ref
        if st.get("target"):
            handles[st["target"]] = ref
    return {"ops": ops}, handles, rn


# ---------------------------------------------------------------------------
# A selected row: rN[i] resolves against the rows rN came back with
# ---------------------------------------------------------------------------


def bind_row(row: dict):
    """What a selected row binds to.

    A CROSSING's fid IS its ``[x, y]`` token, and a ref slot wants the
    POINT rather than the string that spells it -- a token in a name slot
    is grounded as text and refused. Every other row type keeps its fid.

    A TERMINUS binds the same way, off its own ``xy``: it is COMPUTED, so
    nothing on the surface answers to ``end_0``, and the point needs no
    registry.
    """
    fid = row.get("fid")
    if isinstance(fid, str) and fid.startswith("end_"):
        xy = row.get("xy")
        if isinstance(xy, (list, tuple)) and len(xy) == 2:
            return [float(xy[0]), float(xy[1])]
    pt = parse_xy_token(fid) if isinstance(fid, str) else None
    return list(pt) if pt else fid


# ---------------------------------------------------------------------------
# The adapter: a code block -> the plan the loop runs
# ---------------------------------------------------------------------------
# `_run_plan` consumes ONLY the plan dict, so grounding under the name-match
# floor, the analyze execution, flights, the watchlist and the discourse
# writes are all the loop's. A bare NAME string becomes a GROUNDING.

def _is_place_list(v) -> bool:
    """``near=[A, B]`` is TWO PLACES; ``near=[4630, 2965]`` is one point."""
    return (isinstance(v, (list, tuple)) and len(v) >= 2
            and not is_xy(v))


#: The fid families the world mints. Everything else in a place position is
#: a NAME to ground, so this is the whole boundary between "look it up" and
#: "you already have it". Casings are natural ones only -- these tokens are
#: echoed back from what was served, never invented by the model.
_FID_RE = re.compile(r"(?:prior_|detected_|stop_|end_|v)[0-9]{1,4}",
                     re.IGNORECASE)


def _fid_like(tok) -> bool:
    """Is this string a FID rather than a name?"""
    return isinstance(tok, str) and bool(_FID_RE.fullmatch(tok.strip()))


def _plan_of(steps: list[dict], said: str, rn_start: int, bound: dict,
             discourse: dict | None = None) -> tuple[dict, int]:
    """One segment -> the sentence-shaped plan the mission loop can run.

    ``discourse`` is the mission's own discourse dict; the ONE key read
    here is ``widened``, the widening ledger ``more_of`` pages against.
    Read at compile time because the loop replaces that dict on every
    executed plan.
    """
    prog, handles, rn = compile_steps(
        steps, rn_start, known=dict(bound),
        seen=(discourse or {}).get("widened") or {})

    # Bare names in a ref slot are places to ground, one slot each. A NAME
    # is everything the compiler did not itself put there: a row binding
    # left ref_of as {ref, as}, and a SET binding left it as one of these
    # refs.
    live = {op["ref"] for op in prog["ops"]} | {
        v for v in handles.values()
        if isinstance(v, str) and re.fullmatch(r"r\d+", v)}
    grounds: dict = {}

    def _slot(query: str, why: str) -> str:
        if query not in grounds:
            grounds[query] = (f"g{len(grounds)}", why)
        return grounds[query][0]

    for op in prog["ops"]:
        refs = op["args"].get("refs") or []
        for i, v in enumerate(refs):
            if isinstance(v, str) and v not in live and not _fid_like(v):
                refs[i] = {"ref": f"<anchor:{_slot(v, 'anchor')}>", "as": v}

    def _facing(arg):
        """A heading: a compass word (any of geom_compute's eight) rides as
        a plain string, anything else is a place to look TOWARD. The loop
        reads the two apart by type, so the compass word must not be
        wrapped.
        """
        if isinstance(arg, str) and arg.strip().lower() in COMPASS_UNIT:
            return arg.strip().lower()
        return _site(arg, "facing")

    def _site(arg, why: str):
        """An act's PLACE slot: a point, a fid, or a name to ground.

        A POINT rides as ``("xy", [x, y])`` -- a selected crossing's fid IS
        its ``[x, y]`` token. A NAME is grounded exactly as an op's ref slot
        is. WHAT A BINDING HOLDS IS NEVER LOOKED UP: a fid a selection
        resolved came out of a row, and a fid the model typed stays literal.
        """
        if isinstance(arg, str) and arg in handles:
            got = handles[arg]
            if isinstance(got, (list, tuple)) and len(got) == 2:
                return ("xy", [float(got[0]), float(got[1])])
            return ("lit", got if isinstance(got, str) else arg)
        if isinstance(arg, (list, tuple)) and len(arg) == 2:
            return ("xy", [float(arg[0]), float(arg[1])])
        # A QUOTED [x, y] IS A POINT (a crossing's fid is its token): welcome
        # in a relating slot, refused as a COORDINATE in a moving one.
        _pt = parse_xy_token(arg) if isinstance(arg, str) else None
        if _pt:
            if why in ("move", "cover") and not withheld("prior"):
                raise CodeError(act_needs_a_thing(
                    name="go_to/explore",
                    got=f"{_pt[0]:g}, {_pt[1]:g}")
                    + " A crossing you computed is a ROW: index it "
                      "-- r2 = intersection(\"A Road\", \"B St\") then "
                      "go_to(r2[0]).")
            return ("xy", [float(_pt[0]), float(_pt[1])])
        if isinstance(arg, str) and not _fid_like(arg):
            return ("fid", _slot(arg, why))
        return ("lit", arg)

    # A STANCE IS A PREFIX: what it buys (the side columns on every find
    # row, the facing crops and maps) reaches only the steps after it, so a
    # block that stands and finds means the find under that stance.
    plan_steps: list[dict] = [
        {"do": "stage", "at": _site(st["args"]["at"], "stage"),
         "facing": _facing(st["args"]["facing"])}
        for st in steps if st["call"] == "stand"]
    if prog["ops"]:
        plan_steps.append({"do": "analyze", "program": prog})
    for st in steps:
        if st.get("act"):
            arg = (st["args"].get("_pos") or [None])[0]
            if st["call"] == "explore":
                # A `cover` move flies a pass over the thing (the loop
                # takes the stops off its cells) and ends on its last stop.
                # A SET is one place to cover and compiles element-wise;
                # `said_to` carries the model's own spelling for the ack.
                if isinstance(arg, (list, tuple)) \
                        and not is_xy(arg):
                    move = {"do": "move",
                            "to": [_site(a, "cover") for a in arg]}
                else:
                    move = {"do": "move", "to": _site(arg, "cover")}
                move["cover"] = True
                move["said_to"] = (arg if isinstance(arg, str)
                                   else ", ".join(str(a) for a in arg))
            else:
                move = {"do": "move", "to": _site(arg, "move")}
            plan_steps.append(move)
        elif st["call"] == "look":
            pos = st["args"].get("_pos") or [None]
            arg = pos[0]
            ref = handles.get(arg) if isinstance(arg, str) else None
            # A SET look is the badged frame over its rows and names the
            # EXECUTOR's ref, tested by the ref's shape (a fid may start
            # with r).
            if isinstance(arg, (list, tuple)) \
                    and not is_xy(arg):
                # look([a, b, c]) crops EACH element.
                def _look_tok(a):
                    if isinstance(a, str) and a in handles:
                        got = handles[a]
                        if isinstance(got, (list, tuple)) \
                                and len(got) == 2:
                            return format_xy_token(float(got[0]),
                                                   float(got[1]))
                        if isinstance(got, str) \
                                and not re.fullmatch(r"r\d+", got):
                            return got
                        return a
                    if is_xy(a):
                        return format_xy_token(float(a[0]), float(a[1]))
                    return str(a)
                step = {"do": "crops",
                        "fid": [_look_tok(a) for a in arg]}
            else:
                step = ({"do": "crops", "of": ref}
                        if isinstance(ref, str)
                        and re.fullmatch(r"r\d+", ref)
                        else {"do": "crops", "fid": _site(arg, "crops")})
            # The marks ride the SAME step: one crop request carrying its
            # subject and its context, so the renderer's frame decision and
            # its mark projection see one window.
            _marks = st["args"].get("marks")
            if _marks:
                step["mark_fids"] = [_site(m, "marks") for m in _marks]
            plan_steps.append(step)
        elif st["call"] == "show_map":
            around = st["args"].get("around")
            plan_steps.append({"do": "map", "around": _site(around, "map")})
        elif st["call"] == "look_for":
            # THE SEGMENTER CHANNEL: it arms perception and moves nothing.
            # The segmenter reads WORDS, so look_for("parking_lot") and
            # look_for("parking lot") arm the same class.
            classes = [str(c).strip().lower().replace("_", " ")
                       for c in (st["args"].get("_pos") or []) if str(c).strip()]
            near = st["args"].get("at")
            if near is None:
                plan_steps.append(
                    {"do": "watch", "add": [{"class": c, "role": "target"}
                                            for c in classes]})
            else:
                # The near-form is the TARGETED BOOST, and the loop arms the
                # standing watchlist from it too -- one verb, one semantics.
                plan_steps.append({"do": "boost", "concepts": classes,
                                   "where": [_site(near, "boost")]})
    groundings = [{"slot": s, "query": q.lower(), "provenance": "any",
                   "why": w} for q, (s, w) in grounds.items()]
    # THE COLUMN THIS SEGMENT ORDERED BY -- the last score_by's leading
    # emphasis -- so the loop can say when a large anchor made the sort
    # meaningless (every house on a 4 km road is 0 m from it).
    _order_key = None
    for _op in reversed(prog["ops"]):
        if _op["op"] == "score_by":
            _emphs = _op["args"].get("emphasis") or []
            _emph = (_emphs or [None])[0]
            if isinstance(_emph, dict) and _emph.get("key"):
                _order_key = [str(_emph["key"]), int(_emph.get("w") or 1)]
            break
    # The answer ref: the last handle this segment bound to an OP result, or
    # the last thing it was asked to SHOW; the executor serves its envelope
    # and folds the other ops' hints into it.
    last = None
    for st in reversed(steps):
        t = st.get("target") or ""
        if t.startswith("_"):
            # A lifted temp is this block's private wiring and never the
            # question the block asked.
            continue
        got = handles.get(t) if t else st.get("_ref")
        if isinstance(got, str) and re.fullmatch(r"r\d+", got):
            last = got
            break
    return {"_code_shows": {st["_label"]: st.get("_ref")
                            for st in steps if st.get("_label")},
            "said": said, "groundings": groundings, "steps": plan_steps,
            # WHAT THIS SEGMENT ORDERED BY, so the loop can say when the
            # ordering did not separate anything.
            "_code_order": _order_key,
            "last_ref": last,
            "widened": {st["_page"][0]: st["_page"][1] for st in steps
                        if "_page" in st},
            "rn_end": rn}, rn
