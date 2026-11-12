"""The library the model writes against: ``find``, ``look_for``, ``go_to``,
``explore``, ``look``, ``show_map``, ``stand``, ``more_of``,
``intersection``, ``ends_of`` and ``distance``, and the served ``filter`` and
``order`` of a result."""

from __future__ import annotations

from maps.library.block_scan import _MOVE_SAYS
from maps.library.call_arguments import (
    _CallArguments, _Name, _no_such_fid, _plain, _tok)
from maps.library.op_compiler import (
    VIEWPOINT_RE, CodeError, Ref, act_needs_a_thing, compile_filter, compile_find,
    compile_order, fid_like, refuse_metres_in_within)
from maps.library.rows import Row, Rows, _index_it, _one_row, _places, _set_as_list
from maps.query.result_tables import BROAD_HEAD_ROWS
from maps.query.spatial_math import COMPASS_UNIT
from maps.xy_token import is_xy


# One block's measuring budget: a nested loop of distance() calls is one per
# PAIR, millions of mask-to-mask scans at real set sizes.
MEASURE_CALL_CEILING = 20_000


class _LibraryCalls(_CallArguments):
    """The library: each mission call the model writes, compiled onto ops
    (``op_compiler``) or run on one of the loop's verbs
    (:class:`MissionEnv`)."""

    def _call_find(self, query="", *more, nearest=None, near=None,
                   within=None, dwithin=None, color=None,
                   name=None, north_of=None, south_of=None, east_of=None,
                   west_of=None):
        if more:
            raise CodeError(
                'find takes ONE query -- find("89th Street") then '
                'find("Fuchsia Avenue"), or intersection("89th Street", '
                '"Fuchsia Avenue") for where they cross.')
        if name is not None and query == "":
            raise CodeError(f"the name IS the query: find({name!r})")
        q = str(query)
        # A fid is no name to search for: one no row carries is refused.
        if q.startswith(("prior_", "detected_")) and fid_like(q) \
                and self.env.row_of_fid(q) is None:
            raise _no_such_fid(q)
        refuse_metres_in_within(within)
        nearest = self._slot(nearest)
        near = self._slot(near)
        within = self._slot(within)
        cards = self._cards(north=north_of, south=south_of, east=east_of,
                            west=west_of)
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

    _NEAR_MISS_M = 500.0

    def _near_miss_note(self, ref, query) -> None:
        """Note that nothing a find ranked by distance lies near its
        anchor."""
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
            f" Rows of a watched class appear near {anchor} only when a "
            "flight covers it.")})

    def _rows_order(self, items, *, nearest=None, color=None):
        """A distance or a colour, never both: combining the two can rank a
        row that is second on both below worse rows, so the second is refused
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
                "distance, rows.order(color=\"red\") by colour, "
                "rows.order(direction=\"north\") northernmost first, or "
                "rows.order(<function>) for one you compute yourself.")
        # nearest=[A, B] ranks by both distances in one score_by, as find's
        # does.
        nearest = self._slot(nearest)
        items = self._slot(items)
        p = self._program(items, nearest, color)
        return Rows(self._analyze(p, compile_order(
            p, items, nearest=nearest, color=color)), self)

    def _rows_filter(self, items, *, within=None, near=None,
                     dwithin=None, facing=None, direction=None, **cards):
        # Which-arg validation lives in the compiler; every caller has
        # already refused a kwarg outside the set.
        items = self._slot(items)
        within = self._slot(within)
        near = self._slot(near)
        # facing= is a PLACE and rides the slot resolver; direction= is the
        # compass WORD and rides verbatim, which is the whole point of
        # keeping the word out of the kwarg name.
        facing = self._slot(facing)
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
        if not isinstance(x, Rows) or x._spec:
            raise CodeError("more_of pages a RESULT a find or an ordering "
                            "left")
        name = self.said_name(x, "more_of")
        if x._tied():
            x._refuse_tied(name, "has no next block")
        lo = BROAD_HEAD_ROWS + int(self.env.widened.get(x._ref) or 0)
        p = self._program(name)
        # The rows served are NOT the handle -- they are a window of it --
        # so the envelope is labelled with what was asked for rather than
        # with a name that would claim to be the whole set.
        return Rows(self._analyze(
            p, p.window(x._ref, f"[{lo}:{lo + BROAD_HEAD_ROWS}]"),
            shown=f"more_of({name})", page=(x._ref, lo, name)), self)

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
        a, b = _places(a, "distance"), _places(b, "distance")
        for fid, xy in a + b:
            if xy is None and self.env.row_of_fid(fid) is None:
                raise CodeError(
                    f"distance() got {fid!r}, which names no row on this "
                    "surface -- it measures between rows, result sets and "
                    "[x, y] points.")
        return self._geom.distance(a, b)

    def _call_look(self, x, *, marks=None):
        if isinstance(x, (list, tuple, set, frozenset)) and not x:
            raise CodeError(
                "look() got an empty list; there is nothing to look at.")
        # marks= names the rows whose identity bears on the decision; a
        # named road is drawn as its NAME, everything else outlines and
        # badges.
        mk = None
        if marks is not None:
            mk = [self._slot(m) for m in (
                marks if isinstance(marks, (list, tuple)) else [marks])]
        # The marks ride the SAME crop request as its subject, so the
        # renderer's frame decision and its mark projection see one window.
        lead: list = []
        fids = None
        if isinstance(x, Rows) and x._spec:
            # A SLICE is looked at through its window: two verbs, one
            # receipt, and the window is no answer of its own.
            name = self.spelling(x)
            p = self._program(x._ref, mk)
            ref = p.window(x._ref, x._spec)
            self._show_as(ref, name)
        else:
            # A set inside a list is refused under the model's spelling,
            # before _slot would materialize a slice; a string naming a held
            # set is caught once _slot has made it a Ref.
            sets = [a for a in (
                x if isinstance(x, (list, tuple, set, frozenset)) else ())
                if isinstance(a, Rows)]
            if sets:
                held = self.spelling(sets[0])
                if not held:
                    raise CodeError(
                        "look() got an unnamed set inside a list; bind it "
                        "first (s = ...) and name its rows: "
                        "look([s[0], ...])")
            else:
                x = self._slot(x)
                many = isinstance(x, (list, tuple)) and not is_xy(x)
                held = many and next(
                    (a.label for a in x if isinstance(a, Ref)), None)
            if held:
                raise CodeError(
                    f"look({held}) shows a set; inside a list "
                    f"name its rows: look([{held}[0], ...])")
            if many:
                # look([a, b, c]) crops EACH element.
                fids = [self._site(a, "crops") for a in x]
                lead = [(f, "crops") for f in fids]
            elif not isinstance(x, Ref):
                site = self._site(x, "crops")
                lead = [(site, "crops")]
        marks = [(self._site(m, "marks"), "marks") for m in mk or []]
        ties = self._ground(lead + marks)
        toks = [_tok(s, ties) for s, _w in marks] if mk else None
        if isinstance(x, Rows):
            ack = self.env.analyze(p.ops, p.rn, None, None)
            if "error" not in ack:
                ack.update(self.env.crops(ref, None, toks))
        elif isinstance(x, Ref):
            # A SET look is the badged frame over its rows.
            ack = self.env.crops(x.ref, None, toks)
        else:
            ack = self.env.crops(None, [_tok(f, ties) for f in fids]
                                 if fids is not None else _tok(site, ties),
                                 toks)
        self._end(ack)
        return None

    def _call_show_map(self, around=None):
        if around is None:
            raise CodeError(
                "around= got None, which is no place: show_map(around=<place>) "
                "centres on a row, a fid or an [x, y] point.")
        if isinstance(around, Rows):
            raise CodeError(
                self._not_one_place("around", around) + (" -- or use a "
                "name. (A set IS one "
                "place in near=, within= and nearest=, where it means "
                "the gap to its NEAREST member, but there is no one "
                "point to centre on.)" if len(around) else "."))
        around = self._slot(around)
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
        if isinstance(at, Rows):
            raise CodeError(self._not_one_place("at", at) + ".")
        at = self._slot(at)
        if at is None:
            self._act(self.env.watch, [], words)
        else:
            # The placed form is the TARGETED BOOST, and the loop arms the
            # standing watchlist from it too -- one verb, one semantics.
            self._act(self.env.boost, [(self._site(at, "boost"), "boost")],
                      words)
        return None

    def _call_stand(self, *, at, facing="north"):
        for kw, v in (("at", at), ("facing", facing)):
            if isinstance(v, Rows):
                raise CodeError(self._not_one_place(kw, v) + ".")
        at = self._slot(at)
        # A heading: a compass word rides as itself, anything else is a
        # place to look TOWARD.
        word = facing.strip().lower() if isinstance(facing, str) else None
        if word not in COMPASS_UNIT:
            facing = self._slot(facing)
        at_site = self._site(at, "stage")
        f_site = word if word in COMPASS_UNIT else self._site(facing, "facing")
        self._act(self.env.stand, [(at_site, "stage"), (f_site, "facing")])
        st = self._out["stance"]
        h = (st[-1] if isinstance(st, list) else st)["staged"][-1]["handle"]
        self.ns[h] = h
        return h

    def _move(self, call: str, x):
        # A string naming a held set IS that set: go_to("r1") is go_to(r1).
        if isinstance(x, str) and isinstance(dict.get(self.ns, x), Rows):
            x = self.ns[x]
        if isinstance(x, Rows):
            # A SET is one place for explore; go_to lands on one thing (an
            # empty set takes the empty refusal below).
            if call == "go_to" and len(x):
                name = self.spelling(x)
                s = name or "s"
                raise CodeError(
                    f"go_to() takes ONE thing and {name or 'that'} is a set "
                    f"of {len(x)} row(s). {_index_it(name)} -- go_to({_one_row(x, s)})"
                    f" -- or explore({s}) to cover the whole set.")
        elif isinstance(x, (list, tuple, set, frozenset)):
            if isinstance(x, (set, frozenset)):
                x = _set_as_list(x)
            if not x:
                raise CodeError(
                    f"{call}() got an empty list; there is nothing to "
                    f"{_MOVE_SAYS[call]}.")
            if any(e is None for e in x):
                raise CodeError(
                    f"each {call}() place got None, which is no place: the "
                    "name it came from holds no row. Bind a row first.")
            # explore's list is a SET of things -- rows, sets, fids -- as
            # one place to cover; any numeric element is the coordinate
            # hatch.
            things = (not is_xy(x)
                      and all(isinstance(e, (Row, Rows, str)) for e in x))
            ok_set = call == "explore" and things
            # A SET OF THINGS IS NOT A COORDINATE: go_to refuses it as a set.
            if things and not ok_set:
                raise CodeError(
                    f"{call}() takes ONE thing and that is a set of "
                    f"{len(x)}. Index the one you mean -- {call}(<set>[0])"
                    " -- or explore(<set>) to cover all of it.")
            if not ok_set:
                raise CodeError(act_needs_a_thing(
                    name="go_to/explore",
                    got=", ".join(f"{v:g}" if isinstance(v, (int, float))
                                  else str(v) for v in x)))
        for e in (x if isinstance(x, (list, tuple)) else [x]):
            if isinstance(e, Rows) and not len(e):
                raise CodeError(
                    f"{self.spelling(e) or 'that set'} holds 0 row(s) here; "
                    f"there is nothing to {_MOVE_SAYS[call]}.")
        if isinstance(x, str) and VIEWPOINT_RE.fullmatch(x):
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
