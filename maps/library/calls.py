"""The library the model writes against: ``find``, ``look_for``, ``go_to``,
``explore``, ``look``, ``show_map``, ``stand``, ``more_of``,
``intersection``, ``ends_of`` and ``distance``, the served ``filter`` and
``order`` of a result, and how one call runs: its arguments as the compiler
reads them, the grounding that opens it and the ack that closes it."""

from __future__ import annotations

import re

from maps.library.block_scan import MOVE_SAYS
from maps.library.call_arguments import (
    SITE_SLOT,
    MissionEnv,
    Names,
    PlaceName,
    _Stop,
    as_spelled,
    fold,
    grounded_token,
    no_such_fid,
    place_token,
)
from maps.library.op_compiler import (
    CARDINAL_ORDER,
    HANDLE_RE,
    VIEWPOINT_RE,
    Bind,
    CodeError,
    Program,
    Ref,
    act_needs_a_thing,
    compile_filter,
    compile_find,
    compile_order,
    fid_like,
    refuse_metres_in_within,
    rn_spelled,
)
from maps.library.rows import Row, Rows, index_it_remedy, one_row_spelling, places_of, set_as_list
from maps.query.color_words import COLOR_ENUM, normalize_color
from maps.query.result_tables import BROAD_HEAD_ROWS
from maps.query.row_geometry import SurfaceGeometry
from maps.query.spatial_math import COMPASS_UNIT
from maps.xy_token import is_xy

# One block's measuring budget: a nested loop of distance() calls is one per
# PAIR, millions of mask-to-mask scans at real set sizes.
MEASURE_CALL_CEILING = 20_000


def _refuse_unplain_class(cls: str, names: set[str]) -> None:
    """Refuse a look_for class the served text rules out: a proper noun (a
    world-state name written with a capital, ``names`` lowered), or one
    carrying a colour word. The name check runs first: "Green Street" is a
    street, never a colour."""
    words = cls.replace("_", " ").split()
    # ponytail: a lowercase class a feature happens to be named ("car",
    # "pitch") is watched as said, with the receipt's NAMED feature note.
    if cls != cls.lower() and " ".join(words).lower() in names:
        raise CodeError(
            f'look_for takes plain visual classes, no proper nouns: "{cls}" is a proper noun. '
            f'A named place is already a world-state row: find("{cls}").'
        )
    # ponytail: any palette word refuses, so "orange tree" does too; a
    # head-noun parse if a real class ever needs a colour word.
    colour = next((w for w in words if normalize_color(w) in COLOR_ENUM), None)
    if colour:
        raise CodeError(
            f'look_for takes plain visual classes, each naming a THING, never a colour: "{cls}" '
            f'carries the colour "{colour}". A colour ORDERS the rows once they exist: '
            f'find("<class>", color="{colour.lower()}").'
        )


class Library:
    """The library: each mission call the model writes, compiled onto ops
    (``op_compiler``) or run on one of the loop's verbs
    (:class:`MissionEnv`). The sets and rows a call hands back reach their
    served ``filter``, ``order`` and ``distance`` through it."""

    def __init__(self, env: MissionEnv, names: Names):
        self.env = env
        self.names = names
        self.ns = names.ns
        # Footprint geometry BY FID off the live surface, cached until a
        # call ends: a flight folds detections in under it.
        self._geom = SurfaceGeometry(env.row_of_fid)
        self._call_minted: list = []  # handles the last call's flights minted
        self.begin_block({})

    def begin_block(self, out: dict) -> None:
        """A block starts: ``out`` is the receipt its calls fold into."""
        self.out = out
        #: The refs whose envelope the receipt keeps whole.
        self.shown_whole: set = set()
        #: The refs a dedup left unrendered this block -> surface version.
        self.unrendered: dict = {}
        self._noted: set = set()  # keys note_once noted this block
        self._measure_calls = 0
        self._geom.clear()

    def note(self, text: str) -> None:
        """One note on the block's receipt."""
        fold(self.out, {"note": text})

    def note_once(self, key, text: str) -> None:
        """The block's note for ``key``: a repeat of ``key`` notes nothing."""
        if key not in self._noted:
            self._noted.add(key)
            self.note(text)

    def show(self, value, label: str | None) -> None:
        """Serve a set or a row the block showed bare, under ``label``."""
        if isinstance(value, Rows):
            if not value._spec and self.unrendered.get(value._ref) == self.env.version():
                return
            # Key the envelope by the MODEL's spelling ("r1[0:5]"), not the
            # executor's ref the window compiled onto (the window shows it
            # under a label; a nameless set is served whole all the same).
            self.shown_whole.add(self._window(value._ref, value._spec, label, value._ref))
        elif value._src and not re.fullmatch(r"\w+", label):
            # Render the ROW: a one-row window keeps the ordering
            # attribute a bare dict loses, so the model sees the row it
            # asked for rather than a census of one.
            ref, i = value._src
            self._window(ref, f"[{i}:{i + 1}]", label, ref)
        else:
            self.out.setdefault("shown", {})[label] = value._bind

    # -- how one call runs ------------------------------------------------

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
            name = self.names.spelling(v)
            if v._spec:
                v = self._materialize(v)
            return Ref(v._ref, name or self.names.unnamed_label(v._ref))
        if isinstance(v, Row):
            return Bind(v._bind, self.names.name_of(v) or (v.fid or "row"))
        if isinstance(v, str):
            return self._held(v) or v
        if isinstance(v, dict):
            raise CodeError(f"a place is a row, a set, a fid or an [x, y]; got a dict ({v!r:.60}).")
        if isinstance(v, (set, frozenset)):
            v = set_as_list(v)
        if isinstance(v, (list, tuple)) and not is_xy(v):
            # a LIST of things -- order(nearest=[roadA, roadB]) -- maps
            # element-wise.
            out = [self._slot(x) for x in v]
            return out if any(isinstance(x, (Row, Rows)) for x in v) else type(v)(out)
        return v

    def _cards(self, **anchors) -> list:
        """The cardinal cuts given, slotted in the order named, read in
        the compiler's order."""
        got = {d: self._slot(a) for d, a in anchors.items() if a is not None}
        return [(d, got[d]) for d in CARDINAL_ORDER if d in got]

    def _program(self, *args) -> Program:
        """A program for one call: it mints past the rN names the call spells
        and the mission's handles hold."""
        taken = rn_spelled(args) | {
            k
            for k, v in self.ns.items()
            if HANDLE_RE.fullmatch(k)
            and (isinstance(v, Row) or isinstance(v, Rows) and not v._spec)
        }
        live = {v._ref for v in self.ns.values() if isinstance(v, Rows) and not v._spec}
        return Program(self.env.plan_rn[0], taken, live)

    def _ground(self, sites) -> dict:
        """Open a call: ground the NAMES among its ``(site, why)`` pairs ->
        ``{name: top-tie fids}``."""
        why: dict = {}
        for n, w in sites:
            if isinstance(n, PlaceName):
                if not n.strip():
                    raise CodeError(f"{SITE_SLOT[w]} got an empty name, which is no place.")
                why.setdefault(n, w)
        got = self.env.ground(why)
        if "error" in got:
            self._end(got)
        return got["ties"]

    def _analyze(self, p: Program, ref: str, *, shown=None, page=None) -> str:
        """Ground the program's named anchors and run it, ``ref``'s envelope
        the answer (served under ``shown`` when given)."""
        ties = self._ground((PlaceName(n), "anchor") for n in p.anchor_names())
        p.ground(ties)
        ack = self.env.analyze(p.ops, p.rn, ref, page)
        if shown is not None and "error" not in ack:
            # a refused call's ref is minted again by the next program
            self._show_as(ref, shown)
        if "results" not in ack.get("result", {"results": None}):
            # deduped: its note says so, while the surface stands
            self.unrendered[ref] = self.env.version()
        self._end(ack)
        return ref

    def _act(self, verb, sites: list, *args, **kw) -> None:
        """Ground the NAMES among ``sites`` (``(site, why)`` pairs), then run
        the act ``verb`` on their tokens."""
        ties = self._ground(sites)
        self._end(verb(*[grounded_token(s, ties) for s, _w in sites], *args, **kw))

    def _end(self, ack: dict) -> None:
        """Close one call: finish an ack nothing refused, fold it into the
        receipt, bind what its flights minted; a refusal ends the block. A
        call may have changed the surface, so measured geometry is dropped."""
        if "error" not in ack:
            ack = self.env.finish(ack)
        self._geom.clear()
        fold(self.out, ack)
        self._call_minted = self.env.take_minted()
        for h in self._call_minted:
            self.ns[h] = Rows(h, self)
        if "error" in ack:
            raise _Stop()

    def _show_as(self, ref: str, label: str | None) -> None:
        """Serve ``ref``'s envelope under ``label``, kept whole (unnamed when
        ``label`` is None)."""
        if label is not None:
            self.names.shown[ref] = label
        self.shown_whole.add(ref)

    def _window(self, ref: str, spec: str, label: str, *args) -> str:
        """The identity window over ``ref`` (sliced by ``spec``), served
        under ``label``."""
        p = self._program(*args)
        return self._analyze(p, p.window(ref, spec), shown=label)

    def _materialize(self, sliced: Rows) -> Rows:
        """A sliced set becomes a real ref via the identity window, so a
        slice rides every slot a set can (find(near=r1[0:5]) is the union
        of the head, which is what the spelling plainly means)."""
        return Rows(
            self._window(sliced._ref, sliced._spec, self.names.spelling(sliced), sliced._ref), self
        )

    # -- the calls --------------------------------------------------------

    def find(
        self,
        query="",
        *more,
        nearest=None,
        near=None,
        within=None,
        dwithin=None,
        color=None,
        name=None,
        north_of=None,
        south_of=None,
        east_of=None,
        west_of=None,
    ):
        if more:
            raise CodeError(
                'find takes ONE query -- find("89th Street") then '
                'find("Fuchsia Avenue"), or intersection("89th Street", '
                '"Fuchsia Avenue") for where they cross.'
            )
        if name is not None and query == "":
            raise CodeError(f"the name IS the query: find({name!r})")
        q = str(query)
        # A fid is no name to search for: one no row carries is refused.
        if q.startswith(("prior_", "detected_")) and fid_like(q) and self.env.row_of_fid(q) is None:
            raise no_such_fid(q)
        refuse_metres_in_within(within)
        nearest = self._slot(nearest)
        near = self._slot(near)
        within = self._slot(within)
        cards = self._cards(north=north_of, south=south_of, east=east_of, west=west_of)
        p = self._program(q, nearest, near, within, dwithin, color, name, cards)
        ref = compile_find(
            p,
            q,
            name=name,
            near=near,
            nearest=nearest,
            dwithin=dwithin,
            within=within,
            cards=cards,
            color=color,
        )
        self._analyze(p, ref)
        self._near_miss_note(ref, query)
        if q:
            self.names.labels[ref] = f"find({query!r})"
        return Rows(ref, self, label=f"find({query!r})" if q else "")

    _NEAR_MISS_M = 500.0

    def _near_miss_note(self, ref, query) -> None:
        """Note that nothing a find ranked by distance lies near its
        anchor."""
        rows = self.env.rows_of_ref(ref) if ref else []
        if not rows:
            return
        dkey = next(
            (k for k in rows[0] if isinstance(k, str) and k.startswith("distance to ")), None
        )
        if dkey is None:
            return
        vals = [r.get(dkey) for r in rows if isinstance(r.get(dkey), (int, float))]
        if not vals or min(vals) <= self._NEAR_MISS_M:
            return
        anchor = dkey[len("distance to ") :]
        self.note(
            f"nothing found for {str(query)!r} lies near {anchor} -- the "
            f"nearest match is {min(vals):,.0f} m away."
            f" Rows of a watched class appear near {anchor} only when a "
            "flight covers it."
        )

    def rows_order(self, items, *, nearest=None, color=None):
        """A distance or a colour, never both: combining the two can rank a
        row that is second on both below worse rows, so the second is refused
        and named as a cut."""
        if nearest is not None and color is not None:
            raise CodeError(
                "order takes ONE key: a table sorts by one column. "
                "nearest= and color= are two, and combining them ranks "
                "what is mediocre at both above what is right. Cut on one "
                "-- rows.filter(near=<place>, dwithin=<metres>) -- then "
                "order the survivors by the other."
            )
        if nearest is None and color is None:
            raise CodeError(
                "order needs its key: rows.order(nearest=<place>) by "
                'distance, rows.order(color="red") by colour, '
                'rows.order(direction="north") northernmost first, or '
                "rows.order(<function>) for one you compute yourself."
            )
        # nearest=[A, B] ranks by both distances in one score_by, as find's
        # does.
        nearest = self._slot(nearest)
        items = self._slot(items)
        p = self._program(items, nearest, color)
        return Rows(self._analyze(p, compile_order(p, items, nearest=nearest, color=color)), self)

    def rows_filter(
        self, items, *, within=None, near=None, dwithin=None, facing=None, direction=None, **cards
    ):
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
        p = self._program(items, within, near, dwithin, facing, direction, cards)
        return Rows(
            self._analyze(
                p,
                compile_filter(
                    p,
                    items,
                    within=within,
                    near=near,
                    dwithin=dwithin,
                    facing=facing,
                    direction=direction,
                    cards=cards,
                ),
            ),
            self,
        )

    def intersection(self, a, b):
        a, b = self._slot(a), self._slot(b)
        p = self._program(a, b)
        return Rows(
            self._analyze(p, p.add("crossings", {"refs": [p.ref_of(a), p.ref_of(b)]})), self
        )

    def ends_of(self, x):
        x = self._slot(x)
        p = self._program(x)
        return Rows(self._analyze(p, p.add("ends", {"refs": [p.ref_of(x)]})), self)

    def more_of(self, x):
        if not isinstance(x, Rows) or x._spec:
            raise CodeError("more_of pages a RESULT a find or an ordering left")
        name = self.names.own_name(x) or self.names.said_name(x, "more_of")
        if x._tied():
            x._refuse_tied(name, "has no next block")
        lo = BROAD_HEAD_ROWS + int(self.env.widened.get(x._ref) or 0)
        p = self._program(name)
        # The rows served are NOT the handle -- they are a window of it --
        # so the envelope is labelled with what was asked for rather than
        # with a name that would claim to be the whole set.
        return Rows(
            self._analyze(
                p,
                p.window(x._ref, f"[{lo}:{lo + BROAD_HEAD_ROWS}]"),
                shown=f"more_of({name})",
                page=(x._ref, lo, name),
            ),
            self,
        )

    def distance(self, a, b) -> float:
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
                "member."
            )
        a, b = places_of(a, "distance"), places_of(b, "distance")
        for fid, xy in a + b:
            if xy is None and self.env.row_of_fid(fid) is None:
                raise CodeError(
                    f"distance() got {fid!r}, which names no row on this "
                    "surface -- it measures between rows, result sets and "
                    "[x, y] points."
                )
        return self._geom.distance(a, b)

    def look(self, x, *, marks=None):
        if isinstance(x, (list, tuple, set, frozenset)) and not x:
            raise CodeError("look() got an empty list; there is nothing to look at.")
        # marks= names the rows whose identity bears on the decision; a
        # named road is drawn as its NAME, everything else outlines and
        # badges.
        mk = None
        if marks is not None:
            mk = [self._slot(m) for m in (marks if isinstance(marks, (list, tuple)) else [marks])]
        # The marks ride the SAME crop request as its subject, so the
        # renderer's frame decision and its mark projection see one window.
        lead: list = []
        fids = None
        if isinstance(x, Rows) and x._spec:
            # A SLICE is looked at through its window: two verbs, one
            # receipt, and the window is no answer of its own.
            name = self.names.spelling(x)
            p = self._program(x._ref, mk)
            ref = p.window(x._ref, x._spec)
            self._show_as(ref, name)
        else:
            # A set inside a list is refused under the model's spelling,
            # before _slot would materialize a slice; a string naming a held
            # set is caught once _slot has made it a Ref.
            sets = [
                a
                for a in (x if isinstance(x, (list, tuple, set, frozenset)) else ())
                if isinstance(a, Rows)
            ]
            if sets:
                held = self.names.spelling(sets[0])
                if not held:
                    raise CodeError(
                        "look() got an unnamed set inside a list; bind it "
                        "first (s = ...) and name its rows: "
                        "look([s[0], ...])"
                    )
            else:
                x = self._slot(x)
                many = isinstance(x, (list, tuple)) and not is_xy(x)
                held = many and next((a.label for a in x if isinstance(a, Ref)), None)
            if held:
                raise CodeError(
                    f"look({held}) shows a set; inside a list name its rows: look([{held}[0], ...])"
                )
            if many:
                # look([a, b, c]) crops EACH element.
                fids = [place_token(a, "crops") for a in x]
                lead = [(f, "crops") for f in fids]
            elif not isinstance(x, Ref):
                site = place_token(x, "crops")
                lead = [(site, "crops")]
        marks = [(place_token(m, "marks"), "marks") for m in mk or []]
        ties = self._ground(lead + marks)
        toks = [grounded_token(s, ties) for s, _w in marks] if mk else None
        if isinstance(x, Rows):
            ack = self.env.analyze(p.ops, p.rn, None, None)
            if "error" not in ack:
                ack.update(self.env.crops(ref, None, toks))
        elif isinstance(x, Ref):
            # A SET look is the badged frame over its rows.
            ack = self.env.crops(x.ref, None, toks)
        else:
            ack = self.env.crops(
                None,
                [grounded_token(f, ties) for f in fids]
                if fids is not None
                else grounded_token(site, ties),
                toks,
            )
        self._end(ack)
        return None

    def show_map(self, around=None):
        if around is None:
            raise CodeError(
                "around= got None, which is no place: show_map(around=<place>) "
                "centres on a row, a fid or an [x, y] point."
            )
        if isinstance(around, Rows):
            raise CodeError(
                self.names.not_one_place("around", around)
                + (
                    " -- or use a "
                    "name. (A set IS one "
                    "place in near=, within= and nearest=, where it means "
                    "the gap to its NEAREST member, but there is no one "
                    "point to centre on.)"
                    if len(around)
                    else "."
                )
            )
        around = self._slot(around)
        self._act(self.env.map, [(place_token(around, "map"), "map")])
        return None

    def look_for(self, *classes, at=None):
        # at=, not near=: this names WHERE to segment extra.
        if not classes:
            raise CodeError('look_for takes the classes to watch: look_for("house", "pool")')
        if not all(isinstance(c, str) for c in classes):
            raise CodeError('look_for takes class names as strings: look_for("house", "pool")')
        names = self.env.world_names()
        for c in classes:
            _refuse_unplain_class(c, names)
        # THE SEGMENTER CHANNEL: it arms perception and moves nothing. The
        # segmenter reads WORDS, so look_for("parking_lot") and
        # look_for("parking lot") arm the same class.
        words = [w.strip().lower().replace("_", " ") for w in classes if w.strip()]
        refused = self.env.past_cap(words)
        if refused:
            raise CodeError(refused)
        if isinstance(at, Rows):
            raise CodeError(self.names.not_one_place("at", at) + ".")
        at = self._slot(at)
        if at is None:
            self._act(self.env.watch, [], words)
        else:
            # The placed form is the TARGETED BOOST, and the loop arms the
            # standing watchlist from it too -- one verb, one semantics.
            self._act(self.env.boost, [(place_token(at, "boost"), "boost")], words)
        return None

    def stand(self, *, at, facing="north"):
        for kw, v in (("at", at), ("facing", facing)):
            if isinstance(v, Rows):
                raise CodeError(self.names.not_one_place(kw, v) + ".")
        at = self._slot(at)
        # A heading: a compass word rides as itself, anything else is a
        # place to look TOWARD.
        word = facing.strip().lower() if isinstance(facing, str) else None
        if word not in COMPASS_UNIT:
            facing = self._slot(facing)
        at_site = place_token(at, "stage")
        f_site = word if word in COMPASS_UNIT else place_token(facing, "facing")
        self._act(self.env.stand, [(at_site, "stage"), (f_site, "facing")])
        st = self.out["stance"]
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
                name = self.names.spelling(x)
                s = name or "s"
                raise CodeError(
                    f"go_to() takes ONE thing and {name or 'that'} is a set "
                    f"of {len(x)} row(s). {index_it_remedy(name)} -- go_to({one_row_spelling(x, s)})"
                    f" -- or explore({s}) to cover the whole set."
                )
        elif isinstance(x, (list, tuple, set, frozenset)):
            if isinstance(x, (set, frozenset)):
                x = set_as_list(x)
            if not x:
                raise CodeError(
                    f"{call}() got an empty list; there is nothing to {MOVE_SAYS[call]}."
                )
            if any(e is None for e in x):
                raise CodeError(
                    f"each {call}() place got None, which is no place: the "
                    "name it came from holds no row. Bind a row first."
                )
            # explore's list is a SET of things -- rows, sets, fids -- as
            # one place to cover; any numeric element is the coordinate
            # hatch.
            things = not is_xy(x) and all(isinstance(e, (Row, Rows, str)) for e in x)
            ok_set = call == "explore" and things
            # A SET OF THINGS IS NOT A COORDINATE: go_to refuses it as a set.
            if things and not ok_set:
                raise CodeError(
                    f"{call}() takes ONE thing and that is a set of "
                    f"{len(x)}. Index the one you mean -- {call}(<set>[0])"
                    " -- or explore(<set>) to cover all of it."
                )
            if not ok_set:
                raise CodeError(
                    act_needs_a_thing(
                        name="go_to/explore",
                        got=", ".join(
                            f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in x
                        ),
                    )
                )
        for e in x if isinstance(x, (list, tuple)) else [x]:
            if isinstance(e, Rows) and not len(e):
                raise CodeError(
                    f"{self.names.spelling(e) or 'that set'} holds 0 row(s) here; "
                    f"there is nothing to {MOVE_SAYS[call]}."
                )
        if isinstance(x, str) and VIEWPOINT_RE.fullmatch(x):
            # A staged viewpoint is a pose, not a thing on the map: the
            # move wire would resolve it to its stand point, a coordinate.
            raise CodeError(
                f"{x} is a viewpoint you adopted, not a thing on the "
                f"map, and {call}() takes a thing. A stance is for "
                "looking -- its sides ride every find row. To move, "
                f"{call}() a row: r1[0], or a fid off one."
            )
        x = self._slot(x)
        if call == "go_to":
            self._act(self.env.move, [(place_token(x, "move"), "move")])
        else:
            # A `cover` move flies a pass over the thing and ends on its
            # last stop. A SET is one place to cover; a NAME covers its
            # whole top tie group.
            many = isinstance(x, (list, tuple)) and not is_xy(x)
            sites = [place_token(a, "cover") for a in (x if many else [x])]
            said = as_spelled(x)
            ties = self._ground((s, "cover") for s in sites)
            self._end(
                self.env.cover(
                    [
                        ties[s]
                        if isinstance(s, PlaceName) and len(ties[s]) > 1
                        else grounded_token(s, ties)
                        for s in sites
                    ],
                    said if isinstance(said, str) else ", ".join(str(a) for a in said),
                )
            )
        # The arrivals: a flight minted a handle over the rows it added,
        # so `r = explore(x)` binds them.
        return self.ns.get(self._call_minted[-1]) if self._call_minted else None

    def go_to(self, x):
        return self._move("go_to", x)

    def explore(self, x):
        return self._move("explore", x)
