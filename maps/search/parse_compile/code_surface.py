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
from maps.util.xy_token import format_xy_token, parse_xy_token
from maps.search.withhold import withheld

# The executor's helpers are imported INSIDE the functions that use them.
# reasoner_loop imports this module, so a module-level maps import closes
# a cycle -- and a parser has no business needing the executor to load.

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

    ``south`` is satisfied by south, southeast and southwest. Measured on
    the corpus's own ground truth: asking the served column for an EXACT
    word matched the cue on 12.1% of corner-facing cues, because a house
    sits DIAGONALLY from the crossing it is on the corner of, and the
    column reads `southwest` where the cue says `south`. Composition takes
    that cell to 87.9%, and the named-road cell from 87.8% to 89.9% (a cue
    asking a diagonal now matches a measured diagonal). No cell loses.

    The cut is LOOSER by construction -- three of eight winds per cardinal
    against one -- and that is the trade this surface already makes twice:
    `on_side` keeps the row it could not PLACE, `in_or_blank` keeps the row
    it could not MEASURE. Dropping the ground truth is the failure that
    costs points; chaining the two cardinals of a diagonal takes the
    precision back.
    """
    return sorted(w for w in COMPASS_UNIT if cardinal_word in _wind_parts(w))


# order() takes no size key: sorting by footprint_area_m2 measured worse
# than the order the rows arrive in (5 better / 17 worse, p=0.02, where a
# size word is in the cue), because a size adjective in these cues is
# relative to the thing's own kind, not to the map.

# THE MOVING CALLS TAKE A THING, NEVER A COORDINATE. One refusal, because
# a coordinate reaches them by three spellings -- go_to(x, y), go_to([x,
# y]), and a name bound to a literal point one line earlier -- and a
# surface that refuses one of them teaches the other two.
#
# Why the point form is refused at all: it is the escape hatch that lets the
# model answer with an estimate instead of a measurement. Told to find "the
# west side of" a building, a model given points walks coordinates it read
# off a picture, all at one y, while the answer sits in the y it never
# varies, with the segmenter never armed.
# The rows are what carry measured positions; nothing else on this
# surface does.
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
# Every call lands on an op the executor already runs, so this surface adds
# NO execution model. `order` is score_by, `filter` is filter_items,
# `intersection` is crossings -- Python's clothing over the DAG the sentence
# language already compiles onto.
#
# `rN[i]` is the exception and it is deliberate: the executor has no
# take-a-row op, and the sentence language does row selection through
# discourse binding (rule R) rather than through the program. So indexing
# resolves CLIENT-SIDE against the stored rows of that handle, exactly as
# the loop binds `it`. That is the one piece the `dsl` arm lacked.


def _refuse_metres_in_within(v):
    """``within=`` takes a PLACE; a bare number is ``dwithin=``.

    The two names are Shapely's: ``a.within(b)`` is containment and
    ``dwithin(a, b, d)`` is a distance. They were spelled the other way
    round here until this change, so the old ``within=25`` still reads as
    ordinary Python and would otherwise cut against a "place" called 25 --
    a kwarg that does nothing, the hazard this tree has been bitten by
    twice. The two arguments have different TYPES, so the mistake is
    catchable, and it is caught here rather than served as an empty set.
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

    A colour ORDERS and never drops -- the executor rejects a colour filter
    outright, and the language has always been explicit that a hard cut can
    lose the true target.

    It rides the SAME emphasis as the distance terms rather than a second
    score_by, because two sorts in a row is not two constraints: the second
    re-ranks what the first ordered and the first contributes nothing. That
    is the defect that silently dropped the proximity out of
    ``find("house", near="Taylor", south_of="Viburnum Avenue")``.
    """
    if args.get("color"):
        emphasis.append({"key": f"color_{args['color']}", "w": 1})


def compile_steps(steps: list[dict], rn: int = 0,
                  known: dict | None = None,
                  seen: dict | None = None) -> tuple[dict, dict, int]:
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
    # Handles bound by EARLIER segments resolve here too. Without them a
    # segment computes its ordering key from a name it cannot resolve while
    # the ref itself gets substituted later -- the label and the column then
    # disagree and score_by finds nothing to order by.
    handles: dict = dict(known or {})
    # THE TWO NAMESPACES ARE KEPT DISJOINT. The executor's refs are rN and
    # so are the model's handles, and an `as` label rides IN the op's args
    # where the executor's cycle detector reads every string as a possible
    # in-program handle (_consumed_handles). `r3 = r1[1]` then
    # find(near=r3) put "r3" in the compare's args while the compare's own
    # ref was r2 and the score_by's was r3: "reference cycle among ['r2',
    # 'r3']", for a program that is a straight line. So this program never
    # mints an rN any name in play already spells -- which also makes the
    # served receipt's rename unambiguous.
    # EVERY rN the model is using here, argument names included: an
    # UNBOUND r0 is a place to ground, and minting r0 for an op would make
    # the program answer its own question instead.
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
                # the gap to its NEAREST member. Refusing `near=<set>`
                # ("index it -- r2[0] -- or use a name") would be the wrong
                # answer to the question being asked:
                # "the house with a pool in the backyard" is not about pool
                # number one, it is about whichever pool is nearest each
                # house, and indexing throws exactly that away. Fanning out
                # instead would measure every instance separately and die on the
                # executor's token guard the moment either side is large
                # (3,414 houses x 40 pools = 136,560 cells).
                #
                # `items_of` unwraps this back to the bare ref, because the
                # items slot takes a whole result and not a place.
                return {"ref": [got], "as": v}
            # A ROW -- a point or a fid -- and both take the {ref, as}
            # form. The label names the column after the handle the code
            # used; the WRAPPER is what stops a bound fid being read as a
            # bare name, which sent it back through the fuzzy grounding
            # find that produced it ("nothing named 'car_2' is on this
            # surface" once the surface had moved under it).
            return {"ref": list(got) if isinstance(got, (list, tuple))
                    else got, "as": v}
        if isinstance(v, (list, tuple)) and len(v) == 2 \
                and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                        for x in v):
            # AN INLINE POINT TAKES THE SAME WRAPPER, and it must: the
            # executor labels a bare [x, y] 'distance to xy' while
            # key_for sanitizes the literal into 'distance to 4630_2965',
            # so without it the ordering asks for a column the compare never
            # wrote and score_by refuses the whole program, punishing the
            # NATURAL spelling and rewarding the roundabout one (binding the
            # identical point to a name first). Same law as a handle, one
            # door over.
            return {"ref": [float(v[0]), float(v[1])],
                    "as": f"{v[0]:g} {v[1]:g}"}
        if isinstance(v, (list, tuple)):
            # AN ANCHOR MAY BE A SET OF OBJECTS, and this is the one door
            # every slot comes through -- cardinals, within=, facing=,
            # near=, nearest= all resolve here -- so the rule is written
            # once rather than per kwarg.
            #
            # Measured over six 500-episode roots: 156 blocks on 112
            # episodes died on `compare refs element ... must be a bare
            # fid string, an rN result, an [x,y] literal, or {ref, as}`,
            # and the rejected value was almost always a LIST --
            # ['prior_12', 'prior_13'], ['detected_5'], [[x, y]],
            # ['prior_3', 'forest_rows']. That is the top structural
            # refusal on the surface, and `near=[A, B]` had worked the
            # whole time, so the model was reading a slot that accepts a
            # set and generalising correctly.
            #
            # A set folds to its UNION -- ONE ref, ONE column, ONE cut --
            # which is what `ref_of` already means by a whole result
            # (`{"ref": [rN], "as": …}`). Two cuts against two anchors is
            # `.filter(east_of=A).filter(east_of=B)`, which the surface
            # spells and this must not silently become.
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
        # A CROSSING'S FID *IS* ITS [x, y] TOKEN, and the model quotes a fid
        # exactly the way every other fid is quoted (`h = "prior_3123"`).
        # Left a bare string it goes to the grounding find as a NAME and
        # draws `rule N: nothing named "[1058, 3418]" is on this surface` --
        # on 25713026 that killed the surface's own headline chain on 45
        # episodes, one of which burned SIX consecutive blocks on spellings
        # before abandoning an intersection it had already computed
        # correctly. The act slots learned this already (see `_site`); the
        # op ref slots had not.
        _pt = parse_xy_token(v) if isinstance(v, str) else None
        if _pt:
            return {"ref": [float(_pt[0]), float(_pt[1])],
                    "as": f"{_pt[0]:g} {_pt[1]:g}"}
        return v

    def items_of(v):
        """The `items` slot: the SAME lookup, without the label.

        A one-place slot takes ``{ref, as}``; `items` takes a WHOLE result
        and its coercer wants a bare ref or list ("cannot resolve a
        POINT-LIST from {'ref': 'car_1', 'as': 'r2'}"). Same law, two
        shapes, so the two slots ask for it differently.

        A set reaches ``ref_of`` as its UNION -- ``{"ref": ["rN"], "as": …}``
        -- which is what a PLACE slot means by one. Here it means the rows
        themselves, so the one-element wrapper comes back off.
        """
        got = ref_of(v)
        if isinstance(got, dict) and "ref" in got:
            got = got["ref"]
        if (isinstance(got, list) and len(got) == 1
                and isinstance(got[0], str)
                and re.fullmatch(r"r\d+", got[0])):
            return got[0]
        return got

    def key_for(anchor):
        """The column `compare` will write for this anchor.

        Through the executor's OWN naming law, not a guess at it: an `as`
        label is sanitized ("the depot" -> "the_depot"), so a key built by
        string-joining the raw name names a column that does not exist and
        score_by refuses it. One law, called rather than copied.
        """
        from maps.search.analyze_scene.geom_program import _sanitize_token
        r = ref_of(anchor)
        lbl = r["as"] if isinstance(r, dict) and "as" in r else r
        return f"distance to {_sanitize_token(lbl)}"

    def _label_of(anchor):
        from maps.search.analyze_scene.geom_program import _sanitize_token
        r = ref_of(anchor)
        lbl = r["as"] if isinstance(r, dict) and "as" in r else r
        return _sanitize_token(lbl)

    def cardinal_key(direction, anchor):
        """The column `compare` will write for this signed relation.

        Same law as :func:`key_for` -- the executor's own sanitizer, called
        rather than copied, so a key never names a column that does not
        exist and score_by never refuses one it asked for.
        """
        return f"{direction} of {_label_of(anchor)}"

    def _card_ref(anchor):
        """What the compare's ``cardinals`` element addresses.

        The op matches on the ref's fid first and its LABEL second, and the
        label is the one thing available before the surface resolves the
        name, so it is what rides here."""
        return _label_of(anchor)

    def window(src_ref, spec: str = ""):
        """The IDENTITY WINDOW op: the rows a handle already holds, served.

        ``filter_items`` truthy on ``xy`` drops nothing (every served row
        carries an xy), so the op computes nothing and its whole job is to
        make an existing result -- or a slice of one -- the answer of a
        program. That is exactly what the widening form does, and it is
        the only op shape that can spell "the rows I already have": `find`
        re-retrieves and `compare` demands a non-empty refs list.
        """
        ref = mint()
        ops.append({"ref": ref, "op": "filter_items",
                    "args": {"items": f"{src_ref}{spec}", "key": "xy",
                             "op": "truthy"}})
        return ref

    for st in steps:
        call, args = st["call"], st["args"]
        pos = args.get("_pos") or []
        if call == "show":
            # A DISPLAY IS AN OP. A `r1` or `r1[0]` line returns its rows:
            # the legend promises it shows them, and the did-not-bind note
            # tells the model to "Show it first, then index it".
            of, label = args["of"], args["label"]
            name = (of.get("row_of") or of["win_of"]
                    if isinstance(of, dict) else of)
            got = handles.get(name)
            st["_label"] = label
            st["_ref"] = None
            if isinstance(got, str) and re.fullmatch(r"r\d+", got):
                # A WINDOW, never an [i]: an int index yields the bare row
                # dict and the ordering attribute does not survive it, so
                # the envelope downgrades to census + len and the model
                # asked to SEE a row and was handed a count of one. A
                # window keeps its source's order (the executor re-attaches
                # it) and renders the row.
                st["_ref"] = window(
                    got, "" if not isinstance(of, dict)
                    else of["spec"] if "win_of" in of
                    else f"[{of['i']}:{of['i'] + 1}]")
            elif got is not None and not isinstance(of, dict):  # a ROW
                # A handle standing for ONE ROW -- a fid or a point -- names
                # no result to window; the binding IS what there is to show.
                st["_row"] = got
            elif got is not None:
                raise CodeError(
                    f"{name} is one row ({got}), so {label} indexes a row "
                    f"rather than a set: say {name} to see it, or index the "
                    "set it came out of")
            else:
                raise CodeError(
                    f"{label} shows nothing: {name} is not bound. Bind it "
                    f"first -- {name} = find(...) -- or name a handle that is.")
            continue
        if call == "stratum":
            # THE MATCH TIER AS A SET: r1.exact / r1.tagged / r1.fuzzy
            # compile to the executor's own stratum op -- a view over a
            # handle a find left, exactly as a window is.
            src = args["of"]
            got = handles.get(src) if isinstance(src, str) else None
            if not (isinstance(got, str) and re.fullmatch(r"r\d+", got)):
                raise CodeError(
                    f"{src}.{args['tier']} takes a tier of a RESULT, and "
                    f"{src} is "
                    + (f"one row ({got})" if got is not None
                       else "not bound")
                    + ". Name a handle a find left.")
            ref = mint()
            ops.append({"ref": ref, "op": "stratum",
                        "args": {"items": got, "tier": args["tier"]}})
            st["_ref"] = ref
            if st.get("target"):
                handles[st["target"]] = ref
            if not st.get("target") or st["target"].startswith("_"):
                st["_label"] = f"{src}.{args['tier']}"
            continue
        if call == "window":
            # THE TOP-N SLICE. `look(r2[0:20], "...")` is what the model
            # wrote and it was refused; a badged frame over the head of an
            # ordering is the disambiguation the vision channel exists
            # for. Nothing new executes -- the executor windows a handle
            # itself, and this is the identity op the widening form uses.
            src = args["win_of"]
            got = handles.get(src) if isinstance(src, str) else None
            if not (isinstance(got, str) and re.fullmatch(r"r\d+", got)):
                raise CodeError(
                    f"{src}{args['spec']} windows a RESULT, and {src} is "
                    + (f"one row ({got})" if got is not None
                       else "not bound")
                    + ". Name a handle a find or an ordering left.")
            ref = window(got, args["spec"])
            st["_ref"] = ref
            if st.get("target"):
                handles[st["target"]] = ref
            if not st.get("target") or st["target"].startswith("_"):
                st["_label"] = f"{src}{args['spec']}"
            continue
        if call == "bind":
            # A literal binds like a selection resolves: the value IS the
            # handle, and nothing runs.
            handles[st["target"]] = args["value"]
            continue
        if call == "alias":
            # THE SAME RESULT UNDER A SECOND NAME. Nothing runs and no ref
            # is minted: both names resolve to the one the source already
            # holds, so `tanbark_segments.order(...)` and
            # `r1.order(...)` are the same program.
            _of = args["of"]
            if _of not in handles:
                raise CodeError(
                    f"{st['target']} = {_of} names nothing: {_of} is not "
                    f"bound here. Bind it first -- {_of} = find(...) -- or "
                    "name a handle that is.")
            handles[st["target"]] = handles[_of]
            continue
        if st.get("act") or call in ("fly", "look", "show_map", "look_for",
                                     "stand"):
            continue                       # an ACT is a plan step, not an op
        if call == "row_of":
            # A SELECTION IS A SEQUENCING POINT, never an op. Both callers
            # split the block at one (split_at_selections) and resolve it
            # against rows that exist, so reaching the compiler means the
            # split was skipped -- and the {"select": ...} marker that used
            # to be emitted here went to the executor raw: "compare refs
            # object element must be {ref: ...}; got keys ['select']", 12
            # blocks on calcity_code_200.
            raise CodeError(f"{st['target']} = {args['row_of']}"
                            f"[{args['i']}] is a selection, and a selection "
                            "is resolved between programs rather than "
                            "compiled into one")
        if call == "ends_of":
            # THE TWO ENDS OF A LINE, as rows. end0_xy/end1_xy were already
            # SERVED on every linear row and the only thing the model could
            # do with one was type it back as a coordinate, which the moving
            # calls refuse: 44 episodes on 25713026 hit that refusal and
            # scored 13/44, while 106 of 1,023 cues say "end of" and 73 say
            # "cul-de-sac". On calcity::814#808 the served end1_xy IS the
            # ground truth, to within a metre.
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
            _fargs = {"query": pos[0] if pos else args.get("query", "")}
            if args.get("name") is not None:
                _fargs["name"] = args["name"]
            ops.append({"ref": ref, "op": "find", "args": _fargs})
            # ONE WORD, ONE OPERATION. near= ORDERED and dropped nothing,
            # and it reads exactly like a filter -- the model bound its
            # result to `cars_near_poppy`, `houses_near_poppy`, `red_cars`
            # on five straight arms and then treated the constraint as
            # applied. The variable names ARE the false belief. So near=
            # now cuts and needs the gap to cut at, and the ordering is
            # spelled nearest=, the same word .order() already uses.
            if args.get("near") is not None and args.get("dwithin") is None:
                raise CodeError(
                    "near= CUTS and needs the gap to cut at: "
                    "find(<thing>, near=<place>, dwithin=25) keeps what is "
                    "that close and DROPS the rest. To rank without "
                    "dropping, say nearest=<place>.")
            # COLOUR STAYS ON find, deliberately, against the one-word-one-
            # operation tidiness: requiring a second call for it measured 13
            # colour-ordering heads over 2 of 500 episodes, against 440 over
            # 175 of 300 when it was inline. The elicitation cost of moving
            # it is larger than the naming inconsistency it leaves.
            _dwithin = args.get("dwithin")
            if _dwithin is not None:
                # THE CUT WHERE THE MODEL PUTS IT. find(near=X) orders and
                # find(within=X) cuts at contact, so a stated gap on the
                # same call is the one reading left, and the model reaches
                # for it; refusing it burns a turn each time.
                # dwithin= CUTS AGAINST WHICHEVER ANCHOR WAS NAMED. The
                # model wrote find("parking", nearest=[X], dwithin=500) --
                # rank by the gap to X and drop past 500 -- which is
                # coherent, unambiguous, and what anyone would write. There
                # is one anchor either way, so refusing it would be a
                # spelling objection to a correct idea.
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
                cmp_ref = mint()
                ops.append({"ref": cmp_ref, "op": "compare",
                            "args": {"items": ref,
                                     "refs": [ref_of(_anchor)]}})
                out = mint()
                ops.append({"ref": out, "op": "filter_items",
                            "args": {"items": cmp_ref,
                                     "key": key_for(_anchor),
                                     "op": "le", "value": _wv}})
                ref = out
            if args.get("within") is not None:
                # within= CUTS: 0 m contact is a measured geometric fact, and
                # it stays its own compare + filter ahead of every ordering.
                _refuse_metres_in_within(args["within"])
                anchor = args["within"]
                cmp_ref = mint()
                ops.append({"ref": cmp_ref, "op": "compare",
                            "args": {"items": ref,
                                     "refs": [ref_of(anchor)]}})
                out = mint()
                ops.append({"ref": out, "op": "filter_items",
                            "args": {"items": cmp_ref, "key": key_for(anchor),
                                     "op": "le", "value": 0}})
                ref = out
            # THE SIGNED relations. One compare carrying every named anchor
            # (refs is a LIST, so multi-anchor needs no new op), then ONE
            # score_by whose emphasis is every cardinal at +1: further in the
            # asked-for direction ranks higher, and a row the relation cannot
            # place carries no cell, which score_by already treats as the
            # neutral. Nothing is dropped -- the cut is not earned, and a
            # hard directional cut can drop the ground truth.
            #
            # A SUM IS NOT A CONJUNCTION, and this is the honest limit of the
            # ordering. score_by min-max normalizes each key to [0,1] and adds
            # them, so magnitude cannot swamp, but a row that VIOLATES one
            # clause and maximally satisfies the other can still tie the row
            # that satisfies both: two roads 600 m apart, a car 300 m east of
            # the far one scores 1.0 + 0.0 against 0.31 + 0.68 for a car
            # between them. Both columns are SERVED, so the model can read the
            # signs and reject the violator itself; what the sort alone cannot
            # express is "both, at once". Measured, the two-clause shape is 21
            # of 1,023 cues and its intersection contains the target 21/21, so
            # the case this mis-ranks is rare and visible rather than silent.
            # A min-composite emphasis is the fix if the arm shows it costs.
            #
            # ONE compare and ONE score_by for EVERY ordering constraint. Two
            # sorts in a row is not two constraints: the second re-ranks the
            # rows the first ordered and the first contributes nothing, so
            # `find("house", near="Taylor", south_of="Viburnum Avenue")` was
            # ordered by the side alone and the proximity was silently
            # dropped -- the model asked for two things and got one, with no
            # error. Caught reading live blocks off 25713014.
            cards = [(d, args[f"{d}_of"]) for d in _CARDINALS
                     if args.get(f"{d}_of") is not None]
            near_anchor = args.get("nearest")
            if near_anchor is not None or cards:
                # REFS ARE DEDUPED BY LABEL. The executor disambiguates a
                # repeated ref by suffixing its label (`jacaranda_avenue_2`),
                # and the key built here would still name the first -- so
                # score_by would ask for a column that was never written.
                # `find("house", east_of="Jacaranda Avenue",
                # north_of="Jacaranda Avenue")` is the naming of one corner
                # and the model wrote it on the first pass.
                refs: list = []
                seen_labels: set = set()

                def _add_ref(anchor):
                    lab = _label_of(anchor)
                    if lab not in seen_labels:
                        seen_labels.add(lab)
                        refs.append(ref_of(anchor))

                # A CARDINAL CUTS; PROXIMITY ORDERS. Emphasising the signed
                # column ranked by HOW FAR the asked-for way a row is, so
                # rank 0 was the far edge of the map -- median 1,220 m from
                # the anchor, max 5,940 m, which is the map's full width
                # (measured over 54 cardinal tables on 25713026). But 194 of
                # this corpus's 208 cardinal cues mean ADJACENT: "the house
                # east of 87th Street" is the one just east of it, not the
                # one 3 km east. With the served head capped at 24 rows of
                # 2,522, the rows the cue meant were unreachable.
                #
                # So the side is a FILTER (below) and the ordering is
                # NEARNESS to the same anchor. That also retires the
                # sum-is-not-a-conjunction problem: two cardinals are now two
                # cuts and one proximity sort, never a weighted sum of signed
                # magnitudes.
                emphasis: list = []
                seen_keys: set = set()

                def _add_near(anchor):
                    k = key_for(anchor)
                    if k not in seen_keys:
                        seen_keys.add(k)
                        emphasis.append({"key": k, "w": -1})

                # TWO ANCHORS ORDER BETTER THAN ONE, measured: ranking rows
                # by closeness to BOTH features a cue names takes top-24 from
                # 68.7% to 92.9% on the 281 episodes whose cue names two --
                # +6.65 points corpus-wide, held out across three anchor
                # splits at +16 to +28, wins far outnumbering losses in every
                # fold. score_by min-max normalizes each key, so two
                # distances fuse without either magnitude swamping the other.
                #
                # The SENTENCE surface has been able to say this since before
                # any of these runs -- "find <thing> near <place> near
                # <place>." -- and the executor already fuses the keys. Only
                # the CODE surface could not spell it, which is the
                # multi-anchor gap this arm closes.
                #
                # The model is not short of the intent: it writes "the
                # intersection of A and B" 4,994 times and uses it as a
                # FLIGHT TARGET 3,810 times against 291 as a ranking anchor.
                # It names both roads, then throws the pairing away at the
                # moment it ranks.
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
                    cmp_args["cardinals"] = [{"dir": d, "ref": _card_ref(a)}
                                             for d, a in cards]
                cmp_ref = mint()
                ops.append({"ref": cmp_ref, "op": "compare", "args": cmp_args})
                # A SIDE CONSTRAINT CUTS. Ordering alone handed the model
                # the least-wrong row at rank 0 and it read like a hit:
                # find("parking", south_of="California City Middle School")
                # ranked a lot 1,552 m NORTH of the school first, across 163
                # rows none of which was south of it (25713025). A row the
                # geometry PLACES on the other side is not a candidate.
                #
                # `on_side` keeps the rows the column could not place --
                # those are unmeasured, not refuted, and on that same case
                # all 19 real parking rows in the bake abstained. Conjoined
                # constraints chain, which is the AND that a weighted sum of
                # normalized columns could never express.
                for d, a in cards:
                    cut = mint()
                    ops.append({"ref": cut, "op": "filter_items",
                                "args": {"items": cmp_ref,
                                         "key": cardinal_key(d, a),
                                         "op": "on_side"}})
                    cmp_ref = cut
                _add_color(emphasis, args)
                # direction= asks for a COLUMN, so a find that only asks has
                # nothing to rank on. An empty score_by is a sort that
                # sorts by nothing -- emit the compare and stop.
                if emphasis:
                    out = mint()
                    ops.append({"ref": out, "op": "score_by",
                                "args": {"items": cmp_ref,
                                         "emphasis": emphasis}})
                    ref = out
                else:
                    ref = cmp_ref
            elif args.get("color"):
                # COLOUR WITH NO PLACE. Nothing to compare against, so this
                # is the sort alone -- the same score_by order builds,
                # spelled inside the find that already named the class.
                emphasis: list = []
                _add_color(emphasis, args)
                out = mint()
                ops.append({"ref": out, "op": "score_by",
                            "args": {"items": ref, "emphasis": emphasis}})
                ref = out
        elif call == "order":
            items = items_of(pos[0] if pos else args["items"])
            emphasis = []
            anchor = args.get("nearest")
            if anchor is not None:
                # nearest= takes ONE place or a LIST of them, exactly as
                # find's near= does: one compare carrying every anchor,
                # one score_by fusing the normalized distances. The model
                # wrote houses.order(nearest=[roadA, roadB]) against
                # a single-anchor branch and was refused -- the same
                # measured two-anchor win find already carries (+6.65
                # pts corpus-wide) spelled from the other call.
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
            got = handles.get(src) if isinstance(src, str) else None
            if not (isinstance(got, str) and re.fullmatch(r"r\d+", got)):
                raise CodeError(
                    f"more_of({src}) pages a RESULT, and {src} is "
                    + (f"one row ({got})" if got is not None
                       else "not bound")
                    + ". Name a handle a find or an ordering left.")
            from maps.search.analyze_scene.geom_program import (
                _BROAD_HEAD_ROWS)
            lo = _BROAD_HEAD_ROWS + int((seen or {}).get(got) or 0)
            ref = window(got, f"[{lo}:{lo + _BROAD_HEAD_ROWS}]")
            st["_page"] = (got, lo)
            if not st.get("target"):
                # The rows served are NOT the handle -- they are a window
                # of it -- so the envelope is labelled with what was asked
                # for rather than with a name that would claim to be the
                # whole set.
                st["_label"] = f"more_of({src})"
        elif call == "filter":
            items = items_of(pos[0] if pos else args["items"])
            within = args.get("within")
            _refuse_metres_in_within(within)
            near = args.get("near")
            dwithin = args.get("dwithin")
            # A CARDINAL IS A CUT and rides the on_side op find already
            # compiles: the band rule, which ABSTAINS rather than guessing
            # (cardinal_side). A row the geometry cannot place keeps its
            # blank column and is KEPT -- unmeasured is never refuted.
            _cards = [(d, args[f"{d}_of"]) for d in _CARDINALS
                      if args.get(f"{d}_of") is not None]
            # WHICH WAY A THING FACES A PLACE is a CUT, and it needs the
            # anchor: "the south-facing house on Viburnum Avenue" is a
            # question about the house AND the avenue, and it is the
            # largest directional family in the corpus (325 of 1,023
            # cues). Every compare already stamps `facing <ref>` as a
            # compass WORD (geom_program), so this needs no new column --
            # only the cut that reads it, against the cue's own word.
            #
            # The word rides as DATA in direction=, never in the kwarg
            # name: `faces_south=` put it in the name and reached 3 of 45
            # episodes before 8e00c408 deleted it, while the served word
            # column was read 30 times over 6 of 29 and took facing
            # clauses from 15% to 65%.
            #
            # FOUR CARDINALS, and a diagonal is the two of them CHAINED.
            # A thing facing southwest faces south and it faces west, so
            # `direction="south"` keeps it -- the served column still
            # reads all eight winds and the model still reads `southwest`
            # off it; only the CUT composes. Measured on ground truth:
            # exact-word matching answered the cue on 12.1% of
            # corner-facing cues (a house sits diagonally from the
            # crossing it is on the corner of), 87.9% under composition,
            # and the named-road cell went 87.8% -> 89.9%.
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
                # the compiled key IS the key the compare writes: both
                # sides go through _label_of, the executor's own sanitizer.
                #
                # in_or_blank over the winds that HOLD this cardinal, never
                # an equality: south is south, southeast and southwest.
                # And the blank stays kept -- `facing <ref>` is ABSENT at
                # contact, and a house touching the road it fronts is the
                # common case for "the south-facing house on Viburnum
                # Avenue". Unmeasured is never refuted, the same law
                # on_side keeps for the signed columns.
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
                                         {"dir": d, "ref": _card_ref(a)}
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
            # THE GAP IS THE ONLY NEW THING. within= already compiles to a
            # `distance to <anchor>` cut at 0 (contact), so a stated gap is
            # that same cut at a different value -- one number, not a new
            # mechanism, and it inherits the union's nearest-member reading
            # for free.
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
            cmp_ref = mint()
            ops.append({"ref": cmp_ref, "op": "compare",
                        "args": {"items": items, "refs": [ref_of(anchor)]}})
            ref = mint()
            ops.append({"ref": ref, "op": "filter_items",
                        "args": {"items": cmp_ref,
                                 "key": key_for(anchor),
                                 "op": "le", "value": value}})
        elif call == "compare":
            items = items_of(pos[0] if pos else args["items"])
            ref = mint()
            ops.append({"ref": ref, "op": "compare",
                        "args": {"items": items,
                                 "refs": [ref_of(args["to"])]}})
        else:
            raise CodeError(f"there is no {call}()")
        st["_ref"] = ref
        if st.get("target"):
            handles[st["target"]] = ref
    return {"ops": ops}, handles, rn


# ---------------------------------------------------------------------------
# Running a block: selection is a SEQUENCING point
# ---------------------------------------------------------------------------
# `r0[0]` cannot be an op, because which row is row 0 is not known until the
# program that produced r0 has run. So a block compiles to SEGMENTS: execute
# up to the selection, resolve it against the rows that came back, substitute
# the chosen row's fid, continue.
#
# This is not a limitation of the code shape -- it is the executor's own law
# surfacing where it belongs. `intersection("Redwood Boulevard", "85th
# Street")` is THREE crossings on the calcity bake, and `compare` refuses a
# relation to all three ("their centre is at none of them"). The sentence
# surface meets the same wall and answers it with a refusal the model has to
# recover from in prose; here the answer is a subscript.


def bind_row(row: dict):
    """What a selected row binds to.

    A CROSSING's fid IS its ``[x, y]`` token, and a ref slot wants the
    POINT rather than the string that spells it -- a token in a name slot
    is grounded as text and refused. Every other row type keeps its fid.

    A TERMINUS binds the same way, off its own ``xy``. It is COMPUTED, so
    nothing on the surface answers to ``end_0``, and a slot handed that
    name draws "unknown fid 'end_0'; it is not on the surface". Registering
    computed rows with every reader is the general fix and it is not this
    branch's: the sibling that renamed crossings to ``xing_N`` registered
    them with the picture path only, took 252 blocks of that error, and
    lost 17 points. Binding to the POINT needs no registry at all, and it
    is the path a crossing already flies at 96.8%.
    """
    from maps.util.xy_token import parse_xy_token
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
# `_run_plan` consumes ONLY the plan dict. Emit that dict and the whole
# downstream is inherited: grounding resolution under the name-match floor,
# the analyze execution, flights, the watchlist, the discourse writes, and
# every piece of mission law those carry.
#
# A bare NAME string becomes a GROUNDING, exactly as a place
# does, so `find("house", near="Green Street")` resolves its anchor through
# the same plumbing find and the same floor as `find house near Green
# Street.` -- and a name that matches nothing fails with the same rule N.

def _is_place_list(v) -> bool:
    """``near=[A, B]`` is TWO PLACES; ``near=[4630, 2965]`` is one point."""
    return (isinstance(v, (list, tuple)) and len(v) >= 2
            and not _is_point_literal(v))


def _is_point_literal(v) -> bool:
    """A bare [x, y], as opposed to a LIST OF PLACES.

    ``near=[A, B]`` names two anchors; ``near=[4630, 2965]`` is one point.
    Two numbers is the point form and everything else is a place list.
    """
    return (isinstance(v, (list, tuple)) and len(v) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in v))


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

    # Bare names in a ref slot are places to ground, one slot each. What
    # counts as a NAME is everything the compiler did not itself put
    # there: a row binding left ref_of as {ref, as}, and a SET binding
    # left it as one of these refs -- `intersection(r1, "Cross Street")`
    # relates to every way r1 names, which is the escape hatch for a road
    # whose name matches several. Testing `not v.startswith("r")` instead
    # meant a road called "Riverside" was never grounded and reached the
    # executor as a fid, while an unbound r3 reached it as an op ref and
    # came back talking about reference cycles.
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
        """A heading: a compass word rides as a plain string, anything
        else is a place to look TOWARD. The loop reads the two apart by
        type, so the compass word must not be wrapped.

        EIGHT winds, off geom_compute's one table. This read the sentence
        language's four-word `looking` slot, so `stand(facing="southeast")`
        fell through to _site() and was grounded as a lookup for a place
        NAMED "southeast" -- a silent bind, on a word the surface itself
        emits: `facing <ref>` serves all eight. The crop rotation takes any
        unit vector, so four was never the geometry's limit.
        """
        from maps.search.analyze_scene.geom_compute import COMPASS_UNIT
        if isinstance(arg, str) and arg.strip().lower() in COMPASS_UNIT:
            return arg.strip().lower()
        return _site(arg, "facing")

    def _site(arg, why: str):
        """An act's PLACE slot: a point, a fid, or a name to ground.

        A POINT rides as ``("xy", [x, y])``. A selected crossing's fid IS
        its ``[x, y]`` token, so putting it in a ``"lit"`` slot hands the
        loop the TEXT that spells the point and it is grounded as a name:
        `rule N: nothing named "[6076, 4226]" is on this surface`, 28
        blocks over 23 episodes -- the surface's headline chain dying at
        its last step.

        And a NAME is grounded here exactly as an op's ref slot is. The
        act slots took the string verbatim, so `go_to("the depot")` reached
        the flight as a fid nothing on the surface answers to; the
        sentence surface's `go to <place>.` has always resolved a place
        through the plumbing find and the name-match floor, and this is
        that one resolver, reached from the other notation.

        WHAT A BINDING HOLDS IS NEVER LOOKED UP -- ref_of's law, one slot
        kind over. A fid a selection resolved came OUT of a row, so
        sending it back through the fuzzy find that produced it asks for a
        row named after its own fid. Only a bare string the MODEL typed is
        a place, and a fid it typed stays literal: grounding one looks its
        NAME up and finds nothing.
        """
        if isinstance(arg, str) and arg in handles:
            got = handles[arg]
            if isinstance(got, (list, tuple)) and len(got) == 2:
                return ("xy", [float(got[0]), float(got[1])])
            return ("lit", got if isinstance(got, str) else arg)
        if isinstance(arg, (list, tuple)) and len(arg) == 2:
            return ("xy", [float(arg[0]), float(arg[1])])
        # A QUOTED [x, y] IS A POINT, and it reaches here because a crossing's
        # fid IS its token and the model quotes a fid the way it quotes every
        # other one. Left to fall through it is grounded as a NAME and draws
        # `rule N: nothing named "[2707, 5217]" is on this surface` -- the
        # wrong wrongness twice over, because for a RELATING slot the point is
        # welcome and for a MOVING one the objection is not that nothing bears
        # that name, it is that a coordinate is not a thing.
        _pt = parse_xy_token(arg) if isinstance(arg, str) else None
        if _pt:
            if why in ("move", "cover"):
                # The no-coordinate law stands (a point read off a picture is
                # the picture's centre). Refuse it as a COORDINATE so the
                # remedy the model reads is the true one: fly the row, or
                # index the crossing that minted the token.
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

    # A STANCE IS A PREFIX, as it is on the sentence surface: it gates
    # nothing, but what it buys -- the side columns on every find row, the
    # facing crops and maps render up -- is only bought by the steps that
    # come after it. A block that stands and then finds means the find
    # under that stance whichever line the model wrote first.
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
                # THE ONE BIT THAT SEPARATES THE TWO VERBS, and the loop is
                # where it can be honoured: the stops come off the place's
                # own cells, which do not exist at compile time. A `cover`
                # move flies a pass over the thing and ends on its last
                # stop. A SET is one place to cover, so a list compiles
                # element-wise (through _site a len-2 fid list would
                # misparse as an [x, y] point -- the look branch below
                # documents the same hazard); `said_to` carries the
                # model's own spelling so the ack never serves an
                # executor ref.
                if isinstance(arg, (list, tuple)) \
                        and not _is_point_literal(arg):
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
            # A SET look is the badged frame over its rows -- the set-of-marks
            # the whole scene-understanding case rests on -- and it must name
            # the EXECUTOR's ref, not the code's handle for it. The test is
            # the ref SHAPE, not its first letter: a row bound to a fid that
            # happens to start with r (road_1, runway_2) took the set branch
            # and looked at a handle that does not exist.
            if isinstance(arg, (list, tuple)) \
                    and not _is_point_literal(arg):
                # look([a, b, c]) crops EACH element (_say_crops' own
                # list form). Through _site this misparsed: a len-2 fid
                # list became an [x, y] point (float('prior_...') ->
                # ValueError) and every other length stringified into
                # ONE bogus fid refused as "not on the live surface"
                # while its fids were live -- 7 -> 52 blocks the arm the
                # look doc first taught list looks.
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
                    if _is_point_literal(a):
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
            around = st["args"].get("around") or (
                st["args"].get("_pos") or [None])[0]
            plan_steps.append({"do": "map", "around": _site(around, "map")})
        elif st["call"] == "look_for":
            # THE SEGMENTER CHANNEL, and it is what arms perception at
            # all. Nothing on this surface reached it: the last 200-episode
            # run flew 167 flights and returned ZERO detections, because
            # the watchlist was empty on every one of them, while every
            # tool-result footer told the model to call look_for("<class>")
            # -- which it wrote 54 times, unprompted, into a surface that
            # had no such call. FREE: it directs the sensors and moves
            # nothing, so it is legal inside a loop.
            # THE SEGMENTER READS WORDS, so the underscore the model wrote
            # it in does not survive to the wire: look_for("parking_lot")
            # and look_for("parking lot") arm the same class rather than
            # two, one of which SAM would meet as a nonsense open-vocab
            # token (the colour-prefixed class defect, another spelling).
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
    # The answer ref: the last handle this segment bound to an OP result,
    # or the last thing it was asked to SHOW. A handle standing for a fid
    # or a point names no result to read. It matters which: the executor
    # serves the answer ref's envelope and folds the other ops' hints into
    # it, so a display whose ref is not the answer comes back as a hint.
    # THE COLUMN THIS SEGMENT ORDERED BY -- the last score_by's leading
    # emphasis, which is the sort the model is about to read rank 0 off.
    # A large anchor makes that sort meaningless (every house on a 4 km
    # road is 0 m from it), and the loop can only say so if it knows the
    # column.
    _order_key = None
    for _op in reversed(prog["ops"]):
        if _op["op"] == "score_by":
            _emphs = _op["args"].get("emphasis") or []
            _emph = (_emphs or [None])[0]
            if isinstance(_emph, dict) and _emph.get("key"):
                _order_key = [str(_emph["key"]), int(_emph.get("w") or 1)]
            break
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
    return {"_code_binds": {k: v for k, v in handles.items()
                            if isinstance(v, (str, list))},
            "_code_shows": {st["_label"]: (st.get("_ref")
                                           or st.get("_row"))
                            for st in steps if st.get("_label")},
            "said": said, "groundings": groundings,
            # WHAT THIS SEGMENT ORDERED BY, so the loop can say when the
            # ordering did not separate anything. `binds_it` stays False --
            # that is the sentence surface's discourse binding of "it" and
            # this surface has handles instead -- but the ORDERING KEY is
            # the same fact rule R reads, and without it a tie is served
            # silently here while the say surface says so out loud.
            "steps": plan_steps, "binds_it": False,
            "_code_order": _order_key, "bind_key": None,
            "bind_w": None, "new_anchor": None, "new_anchor_el": None,
            "last_ref": last, "crossings": {},
            "widened": {st["_page"][0]: st["_page"][1] for st in steps
                        if "_page" in st},
            "rn_end": rn}, rn
