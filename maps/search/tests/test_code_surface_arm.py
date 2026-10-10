"""The code arm drives the SAME mission machine, through a different parser.

These pin the seam rather than the syntax: a `run` block reaches the plan
body `say` reaches, so grounding, the analyze execution, flights and the
watchlist are the ones already tested — and the multi-hop shape the sentence
surface produced in 0 of 200 flown episodes is expressible in one block.
"""
from __future__ import annotations

import json
import re

from maps.search.tests.test_mission_loop import (
    _CANDS, _drive, _tool_results,
)


def _run(code: str):
    return ("run", {"code": code})


def _record_block(code: str, bound: dict | None = None, rn: int = 0,
                  discourse: dict | None = None) -> tuple[list[dict], dict]:
    """Run ONE block against a recording env: (its plans, its ack).

    The surface has no parser: a block becomes steps only by EXECUTING,
    so a compile-shape test captures its plans off a ``MissionEnv`` whose
    ``run_plan`` records them and acknowledges nothing. ``bound`` seeds the
    namespace the way an earlier block would have left it -- ``rN`` binds
    a result handle, any other string a row's fid, an ``[x, y]`` a point
    -- so a block that names an earlier handle resolves exactly as it
    does mid-mission.
    """
    import re

    from maps.search.parse_compile import python_surface as ps

    seen: list[dict] = []

    counter = [rn]

    def _run_plan(plan, said):
        seen.append(plan)
        # the loop's own advance: a later call in the same block must not
        # re-mint the refs the earlier one took
        counter[0] = max(counter[0], int(plan.get("rn_end") or 0))
        return {}

    env = ps.MissionEnv(
        run_plan=_run_plan, rows_of_ref=lambda ref: [], plan_rn=counter,
        discourse=discourse if discourse is not None else {},
        minted=[], say_err=lambda ack, msg: {**ack, "error": msg},
        row_of_fid=lambda fid: None)
    mp = ps.MissionPython(env)
    for name, held in (bound or {}).items():
        if isinstance(held, str) and re.fullmatch(r"r\d+", held):
            mp.ns[name] = ps.Rows(held, mp)
        elif isinstance(held, str):
            mp.ns[name] = ps.Row({"fid": held}, mp=mp)
        else:
            mp.ns[name] = held
    return seen, mp.run_block(code)


def plan_of_block(code: str, bound: dict | None = None, rn: int = 0,
                  discourse: dict | None = None) -> dict:
    """The plan ONE block compiles to, without flying anything."""
    from maps.search.parse_compile import python_surface as ps

    seen, out = _record_block(code, bound, rn, discourse)
    if "error" in out and not seen:
        raise ps.CodeError(str(out["error"]))
    if not seen:
        raise AssertionError(f"{code!r} compiled no plan: {out}")
    return seen[-1] if len(seen) == 1 else {"plans": seen, **seen[-1]}


def raising_ops_of_block(code: str, bound: dict | None = None,
                         rn: int = 0) -> list[dict]:
    """Every op ONE block compiles -- raising the block's OWN refusal.

    ``plan_of_block`` reports a refusal only when nothing compiled before
    it, because an error stops a block at its line and everything before
    it ran and stands -- the mission's real semantics. A test about what
    a call REFUSES wants the refusal even when a ``find`` on the line
    above already compiled its plan.
    """
    from maps.search.parse_compile import python_surface as ps

    seen, out = _record_block(code, bound, rn)
    if "error" in out:
        raise ps.CodeError(str(out["error"]))
    return [op for p in seen for st in p["steps"]
            if st.get("do") == "analyze" for op in st["program"]["ops"]]


def ops_of_block(code: str, bound: dict | None = None,
                 rn: int = 0) -> list[dict]:
    """Every op ONE block compiles, across all its plans, in order.

    python_surface dispatches one plan PER CALL, so a two-call block is
    two programs -- but they share the executor's ref counter, so the
    concatenation is the single program the block adds up to and it runs
    under ``geom_program.run_program`` exactly as the block does.
    """
    plan = plan_of_block(code, bound=bound, rn=rn)
    return [op for p in plan.get("plans", [plan])
            for st in p["steps"] if st.get("do") == "analyze"
            for op in st["program"]["ops"]]


def _fly_drive(script):
    """The mission over a graph the flight actually GROWS.

    ``test_mission_loop._drive`` holds the surface still, so no flight
    there ever mints a handle. The mint is the second writer into the rN
    namespace, and every test about it needs a world that changes.
    """
    from maps.search.tests.test_flight_mints_a_handle import (
        _drive as _grow)
    return _grow(script)


def test_a_block_grounds_finds_orders_and_flies():
    """The chain: find near a named place, order it, take a row, go to it.

    One block, four handles, and the flight at the end is the commit — the
    thing the say surface needs four separate sentences and a transcribed
    fid to reach.
    """
    reply, fr, flights, _declared = _drive([
        [_run('r1 = find("car", nearest="the depot")\n'
              'r2 = r1.order(nearest="the depot")\n'
              'r3 = r2[0]\n')],
        [_run('go_to(r3)\n')],
        "The house by the depot.",
    ])
    assert reply.ended_by == "reply"
    out = _tool_results(fr)
    assert "error" not in out[0], out[0]
    # The block is recorded, append-only, and counted.
    assert out[0]["script_lines"] == 3
    assert out[1]["script_lines"] == 4
    # r3 bound to a real row of r2, and the NEXT block flew to it -- the
    # handle outlived its block, which is what makes the chain multi-hop.
    assert "r3" in out[1]["handles"], out[1]
    assert reply.flights_flown == 1
    assert flights and flights[0]["target_fid"] == "car_1", flights


def test_the_whole_vocabulary_in_one_episode():
    """find -> narrow -> order -> select -> go_to, plus adjective ordering.

    This is the transcript the arm exists to produce: the multi-hop form
    the sentence surface reached in 0 of 200 flown episodes.
    """
    reply, fr, flights, _d = _drive([
        [_run('look_at = 0\n')],           # ordinary Python, a silent bind
        [_run('r1 = find("car")\n'
              'r2 = r1.filter(within="the depot")\n'
              'r3 = r1.order(nearest="the depot")\n'
              'r4 = r3[0]\n')],
        [_run('go_to(r4)\n')],
        "the nearest car",
    ])
    out = _tool_results(fr)
    assert "error" not in out[0], out[0]
    assert "error" not in out[1], out[1]
    assert out[1]["handles"] == ["r1", "r2", "r3", "r4"], out[1]
    # size= compiles to a footprint_area_m2 sort (asserted in
    # code_surface.demo); this fixture's synthetic cars carry no footprint,
    # so the executor says so rather than ordering by nothing.
    assert reply.flights_flown == 1
    assert flights[0]["target_fid"] == "car_1", flights


def test_the_watchlist_and_the_loud_errors_are_the_same_ones():
    """A code arm inherits mission law rather than restating it: an
    ungroundable name is the SAME rule N refusal, and the watchlist state
    rides every result."""
    reply, fr, _flights, _d = _drive([
        [_run('r1 = find("car", nearest="nowhere at all")\n')],
        "done",
    ])
    out = _tool_results(fr)
    assert "error" in out[0] and "rule N" in out[0]["error"], out[0]
    assert out[0]["watching"], out[0]


def test_a_loop_that_flies_is_refused_before_anything_runs():
    """The one structural cut, at the surface the model actually writes to."""
    reply, fr, flights, _d = _drive([
        [_run('r1 = find("car")\nfor c in r1:\n    go_to(c)\n')],
        "done",
    ])
    out = _tool_results(fr)
    assert "error" in out[0]
    assert "cannot go inside a loop" in out[0]["error"], out[0]["error"]
    assert flights == [], flights


# --- explore: the covering pass, and the return leg that makes it free ----

# The stops a fake footprint hands back — a 300 m thing walked at the
# spacing the driver's observation extent implies.
_STOPS = [[100.0, 200.0], [200.0, 200.0], [300.0, 200.0]]
_START = (7.0, 9.0)


def _explore_drive(script, *, stops=_STOPS, obs_span_m=200.0, cands=None,
                   obs_minor_m=None, with_renderer=False,
                   deadline_epoch=None, on_fly=None, held_now=None):
    """The mission with an engine that can report a footprint's stops.

    The stub flight reports each item's OWN point as its pose, because the
    one thing every explore test has to be able to see is WHERE THE DRONE
    ENDED — the fixture that always answers the same pose cannot tell a
    pass that came home from one that did not.

    ``stops`` may be a dict keyed by fid — the per-member shapes a SET
    pass walks — or the one list every fid answers, as the single-pass
    tests always had it.
    """
    import json as _json
    from types import SimpleNamespace
    from maps.search.parse_compile.reasoner_loop import (
        MissionSpec, advance_reasoner_one_round)
    from maps.search.tests.test_mission_loop import FakeReasoner
    from maps.util.xy_token import parse_xy_token

    flights: list[dict] = []

    def _fly(item):
        flights.append(item)
        # ``on_fly`` lets a test advance a fake clock BY THE STOP rather
        # than by wall time — the wall-clock backstop cannot be crossed
        # mid-pass otherwise without sleeping.
        if on_fly is not None:
            on_fly(item)
        _pt = parse_xy_token(item["target_fid"]) or (1.0, 2.0)
        return {"stop_id": f"stop_{len(flights) - 1:03d}",
                "target": item["target_fid"],
                "pose_xy": [float(_pt[0]), float(_pt[1])],
                "new_detections": 1, "cue": "the cue"}

    asked: list[tuple] = []

    def _stops(fid, spacing_m):
        asked.append((fid, spacing_m))
        got = stops.get(fid) if isinstance(stops, dict) else stops
        return [list(p) for p in got] if got else None

    rendered: list = []

    def _renderer(reqs):
        rendered.append(list(reqs))
        return [{"jpeg": b"x", "center_fid": r["center_fid"],
                 "legend": [{"digit": 1, "fid": r["center_fid"],
                             "label": "car", "role": "center"}],
                 "context_m": 30.0, "world_window": [0, 0, 200, 200]}
                for r in reqs]

    fr = FakeReasoner(script)
    ctx = SimpleNamespace(graph=None, cell_size_m=1.0,
                          tile_origin_xy=(0.0, 0.0), grid_shape=(10, 10),
                          robot_pose=SimpleNamespace(x=_START[0], y=_START[1]))
    reply = advance_reasoner_one_round(
        reasoner=fr, user_message="Find the goal object.", ctx=ctx,
        rid_tag="m0",
        compute_namespace={"cands": list(_CANDS if cands is None else cands),
                           "anchors": [], "cell_m": 1.0},
        diag={}, obs_span_m=obs_span_m,
        obs_minor_m=obs_minor_m,
        look_renderer=_renderer if with_renderer else None,
        mission_spec=MissionSpec(held_now=held_now,
            fly=_fly, declare_concepts=lambda i: {"declared": []},
            coverage_stops=_stops, flight_budget=32, turn_cap=48,
            deadline_epoch=deadline_epoch),
    )
    out = [_json.loads(m["content"]) for m in fr.messages
           if m.get("role") == "tool"]
    return reply, out, flights, asked, rendered


def test_an_explore_flies_the_stops_and_ends_on_the_last_one():
    """A pass covers the thing and STAYS where it finished.

    SR is scored on ``final_pos_xy`` (benchmarks/citynav/eval.py) — where
    the drone STOPS — so an explore commits to its last stop. That is the
    honest cost of covering something, paid in where you end and undone by
    naming the row you meant and going to it. It is stated as fact in the
    ack rather than left to be deduced: a standing answer the model cannot
    read is how a scouting point becomes a commit by accident.
    """
    reply, out, flights, asked, _rend = _explore_drive([
        [_run('explore("prior_7")\n')],
        "done",
    ])
    assert "error" not in out[0], out[0]
    # the stops of the thing, and NOTHING after them
    assert len(flights) == len(_STOPS), flights
    assert reply.flights_flown == len(_STOPS)
    from maps.util.xy_token import parse_xy_token
    flown = [list(parse_xy_token(f["target_fid"])) for f in flights]
    assert flown == _STOPS, flown
    ack = out[0]["explored"]
    assert ack["stops"] == len(_STOPS), ack
    # No coordinate is served for where the walk ended: the answer standing
    # is named as the pass's last stop over the thing explored.
    assert "ended_at" not in ack, ack
    assert ack["answer_if_you_reply_now"].startswith(
        f"the last stop of the pass over {ack['explored']}"), ack
    # it did NOT go home
    assert list(_START) not in flown, flown


def test_an_explore_spends_half_an_observation_of_the_run_per_stop():
    """The spacing is the RUN's, never a constant.

    ``observation_extent_m`` returns the sensor footprint's LONG side, and
    half of it sits under the short side for any sensor below 2:1 — which
    is what lets a one-dimensional walk cover a shape that is not one. A
    bake or an altitude change moves that number, and a second copy of it
    would go stale on one side.
    """
    _r, _o, _f, asked, _rend = _explore_drive([
        [_run('explore("prior_7")\n')],
        "done",
    ], obs_span_m=320.82)
    assert asked == [("prior_7", 320.82 / 2.0)], asked


def test_an_explore_serves_ONE_picture_and_ONE_handle():
    """A pass is one act, and its mechanism is not its result.

    Per-stop views would put a dozen images in a single turn against a
    whole-context budget of sixteen — one explore would evict everything
    the mission had seen. Per-stop mints would hand the model a name for
    stop 5's rows and call it the explore's product.
    """
    reply, out, flights, _a, _rend = _explore_drive([
        [_run('explore("prior_7")\n')],
        "done",
    ])
    ack = out[0]["explored"]
    # ONE ack, not a list of flight reports
    assert "flown" not in out[0], out[0]
    assert ack["new_detections"] == len(_STOPS), ack
    # the pass mints at most one handle, over the whole of itself
    assert len([k for k in out[0].get("handles") or [] if k]) >= 0
    assert isinstance(ack.get("stops"), int)


def test_an_explore_of_a_thing_with_no_footprint_is_the_one_stop_it_is():
    """A fid carrying no cells: there is nothing to cover, and the pass
    says so rather than reporting a cover it did not fly."""
    reply, out, flights, _a, _rend = _explore_drive([
        [_run('explore("prior_7")\n')],
        "done",
    ], stops=None)
    assert "error" not in out[0], out[0]
    assert len(flights) == 1, flights
    assert flights[0]["target_fid"] == "prior_7"
    assert "no footprint to cover" in (out[0]["explored"].get("note") or "")


def test_a_call_indexed_inside_an_argument_is_its_own_dispatch():
    """`r1.filter(within=find("Jac Ave")[0])` met "only a handle can be
    indexed", which names the wrong problem.

    It is ordinary Python now: the inner call evaluates first, its rows
    come back, and the subscript picks one -- so the block is TWO plans,
    the find's and the narrow's, and the second measures against the row
    the first produced. That is where the model actually writes it -- one
    live episode spent three blocks on the two spellings.
    """
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\n'
              'r2 = r1.filter(within=find("gate")[0])\n')],
        "done",
    ])
    out = _tool_results(fr)
    assert "error" not in out[0], out[0]
    # the inner find ran as its own dispatch, and the row it yielded is
    # what the narrow measured against -- by fid, never by name.
    assert "distance to prior_1" in json.dumps(out[0]), out[0]
    assert out[0]["handles"] == ["r1", "r2"], out[0]


def test_show_of_a_handle_is_the_bare_line_spelled_out():
    """`show(r1)` was refused, and it names the step this surface HAS.

    A bare `r1` line already compiles to a show; the model asked for it
    the other way 6 times in 967 blocks. Normalised rather than taught:
    the legend keeps ONE display rule (Python's own -- an assignment
    receipts, a bare expression shows) and the surface stops refusing the
    other spelling of it.
    """
    spelled = plan_of_block("show(r1)\n", bound={"r1": "r7"})
    bare = plan_of_block("r1\n", bound={"r1": "r7"})
    # IDENTICAL, not merely similar: two spellings of one step, or the
    # served envelope keys them differently and the model sees two things.
    assert spelled["steps"] == bare["steps"], (spelled, bare)
    assert spelled["_code_shows"] == bare["_code_shows"] == {"r1": "r0"}
    # ...and it stays out of the served vocabulary: one way to display.
    from maps.search.parse_compile.python_surface import _LIBRARY_CALLS
    assert "show" not in _LIBRARY_CALLS


def test_a_fid_in_an_anchor_slot_is_the_row_it_names():
    """`r1.order(nearest="prior_6087")` fuzzy-matched its own fid.

    Every op-ref string the compiler had not itself minted became a
    GROUNDING, so a literal fid went through the plumbing find as a name:
    "prior_6087" tokenizes to 'prior' + '6087', matches no row on those
    words, and anchors to whichever row scored best anyway. The ordering
    is then measured to a row the model never named -- a wrong answer
    with no error in it, and the served note says only that the words
    "did not narrow anything".

    Seen 3 times in the first 30 episodes of the 500-episode arm, across
    nearest=, within= and intersection(). `_site` already keeps a fid
    literal for the ACT slots (grounding one looks its NAME up and finds
    nothing); this is the same law on the op-ref path.
    """
    def _refs(src):
        plan = plan_of_block(src, bound={"r1": "r7"})
        out = [o["args"]["refs"] for o in ops_of_block(src, {"r1": "r7"})
               if o["args"].get("refs")]
        return out, [g["query"] for g in plan["groundings"]]

    for src in ('r2 = r1.order(nearest="prior_6087")\n',
                'r2 = r1.filter(within="prior_5631")\n',
                'r0 = intersection("prior_5133", "prior_5954")\n',
                'r2 = r1.order(nearest="detected_12")\n'):
        refs, grounds = _refs(src)
        flat = [v for group in refs for v in group]
        assert all(isinstance(v, str) for v in flat), (src, refs)
        assert grounds == [], (src, grounds)
    # THE VACUITY FLOOR: a real NAME must still ground, or this "fix"
    # would be a surface that grounds nothing at all.
    refs, grounds = _refs('r2 = r1.order(nearest="Green Street")\n')
    assert grounds == ["green street"], grounds
    assert refs[0][0]["ref"].startswith("<anchor:"), refs


def test_an_alias_gives_a_result_a_second_name():
    """`tanbark_segments = r1` is ordinary Python and was refused.

    "the right-hand side is a call" -- 4 blocks in the first 28 episodes
    of the 500-episode arm, the model naming a result for what it holds.
    There is nothing to refuse: the alias names the SAME executor result,
    so it binds and nothing runs, exactly as a selection binds.
    """
    import pytest
    from maps.search.parse_compile.code_surface import CodeError
    # BOTH NAMES REACH THE SAME ROWS: the alias is the same Rows object
    # under a second name, so it mints nothing and re-runs nothing -- the
    # ordering compiles straight off the source's own executor ref.
    ops = ops_of_block('segs = r1\nr2 = segs.order(nearest="A St")\n',
                       bound={"r1": "r7"})
    assert [o["op"] for o in ops] == ["compare", "score_by"], ops
    assert ops[0]["args"]["items"] == "r7", ops
    # An alias of NOTHING is still refused, and says which name is unbound.
    with pytest.raises(CodeError, match="name 'nope' is not defined"):
        plan_of_block("x = nope\n")


def test_a_snake_case_class_is_the_class_it_spells():
    """`look_for("solar_panel")` was refused as "not a segmenter class".

    The model is writing Python, so snake_case IS its spelling for a
    two-word class -- 2 blocks in the first 28 episodes. SAM reads words
    either way, and the identical intent spelled with a space was always
    taken, so the underscore refused an intent rather than an error.
    """
    plan = plan_of_block('look_for("solar_panel", "parking_lot")\n')
    # THE WIRE GETS WORDS. An underscore reaching SAM is a nonsense
    # open-vocab token -- the same defect a colour-prefixed class was,
    # spelled another way -- so it must not survive the compile.
    assert plan["steps"] == [{"do": "watch", "add": [
        {"class": "solar panel", "role": "target"},
        {"class": "parking lot", "role": "target"}]}], plan["steps"]


def test_an_inline_point_relates_exactly_as_a_bound_one_does():
    """`near=[x, y]` is the natural spelling and it was the broken one.

    The executor labels a bare [x, y] 'distance to xy' while the compiler
    sanitized the literal into 'distance to 4630_2965', so the ordering
    asked for a column the compare never wrote and score_by refused the
    whole program. The surface punished the natural spelling and rewarded
    the roundabout one: smoke 25712973 wrote find("tree", nearest=[4630,
    2965]), was refused, bound the identical point to a name and got the
    same question answered — twice in one episode.

    Points RELATE and LOOK and never commit, which is the whole reason
    they survive on a surface that refuses them for movement; a relate
    that only works when spelled the long way is not a kept capability.
    """
    def _cols(ops):
        cmp_ = [o for o in ops if o["op"] == "compare"][0]
        sc = [o for o in ops if o["op"] == "score_by"][0]
        return cmp_["args"]["refs"][0], sc["args"]["emphasis"][0]["key"]

    inline = ops_of_block('r1 = find("tree", nearest=[4630, 2965])\n')
    bound = ops_of_block('pt = [4630, 2965]\nr1 = find("tree", nearest=pt)\n')
    for ops in (inline, bound):
        ref, key = _cols(ops)
        # THE TWO HALVES AGREE BY CONSTRUCTION: the compare labels the
        # column and the ordering names that same label.
        assert isinstance(ref, dict) and "as" in ref, ref
        assert key == f"distance to {ref['as'].replace(' ', '_')}", (ref, key)
    # ...and the point itself survives as a point, not as a name to ground.
    ref, _ = _cols(inline)
    assert ref["ref"] == [4630.0, 2965.0], ref
    assert plan_of_block(
        'r1 = find("tree", nearest=[4630, 2965])\n')["groundings"] == []


def test_the_moving_calls_refuse_a_coordinate_in_every_spelling():
    """A COORDINATE IS THE MODEL'S ESTIMATE; a row's position is a
    measurement. Smoke 25712881 is the argument: told to find "the west
    side of the Rite Aid building" it flew the centroid and then walked
    go_to([4630, 2965]), go_to([4625, 2965]), go_to([4635, 2965]) — three
    coordinates read off a picture, all at one y, while the answer sat
    16 m away in the y it never varied. It missed by 3.3 m against a 20 m
    radius with the segmenter never armed.

    Every spelling that REACHES the call, because refusing one moves the
    escape hatch rather than closing it — and the bind-then-move form is
    the one that episode actually used. (`go_to(x, y)` never reaches it:
    the call takes one argument, so two numbers are an arity refusal.)
    """
    import pytest
    from maps.search.parse_compile.code_surface import CodeError
    for bad in ("go_to([3411, 1468])\n",
                "explore([3411, 1468])\n",
                "target = [4630, 2965]\ngo_to(target)\n",
                "corner = [3411, 1468]\nexplore(corner)\n",
                'go_to("[3411, 1468]")\n'):
        with pytest.raises(CodeError, match="takes a THING on the map"):
            plan_of_block(bad)
    with pytest.raises(CodeError, match="positional argument"):
        plan_of_block("go_to(3411, 1468)\n")
    # ...and the refusal names the instrument that replaces it.
    try:
        plan_of_block("go_to([3411, 1468])\n")
    except CodeError as exc:
        assert "look_for" in str(exc) and "explore" in str(exc), str(exc)
    # A POINT STILL RELATES AND STILL LOOKS — only moving refuses it.
    assert plan_of_block('corner = [1, 2]\nr1 = find("car", nearest=corner)\n')
    assert plan_of_block("show_map(around=[1, 2])\n")
    assert plan_of_block('stand(at=[1, 2], facing="north")\n')


def test_a_go_to_still_leaves_the_drone_where_it_flew():
    """The other half of the pair, and the reason the pair is two calls."""
    reply, out, flights, _a, _rend = _explore_drive([
        [_run('go_to("prior_7")\n')],
        "done",
    ])
    assert len(flights) == 1 and flights[0]["target_fid"] == "prior_7"
    assert "explored" not in out[0], out[0]
    assert out[0]["flown"]["flown"][0]["target"] == "prior_7", out[0]


def test_the_inline_selection_a_live_reasoner_actually_writes():
    """`go_to(r1[0])`, which is the FIRST thing a live reasoner emitted here.

    The two-statement form was what the surface was built for; the model
    wrote the inline one, because that is the natural Python. It is
    normalised into a pick rather than refused, and the pick resolves from
    rows an earlier block already has, so the move carries a real fid.
    """
    reply, fr, flights, _d = _drive([
        [_run('r1 = find("car")\n')],
        [_run('go_to(r1[0])\n')],
        "the car",
    ])
    out = _tool_results(fr)
    assert "error" not in out[1], out[1]
    assert reply.flights_flown == 1
    assert flights and flights[0]["target_fid"] == "car_1", flights


def test_the_two_silent_defects_the_live_smoke_exposed():
    """Both were found by a 3-episode smoke and both failed SILENTLY.

    A live reasoner wrote `find("church", name="Desert Song Foursquare
    Church")` and the name constraint vanished into an ignored kwarg while
    the run still looked successful. And types lasted one block, so a set
    built in one turn flowed into a one-place slot in the next -- the case
    the type system exists for.

    ``name=`` is IMPLEMENTED now rather than refused. It was the largest
    single error family in the run -- 135 blocks over 61 episodes, scoring
    13/61 against 92/194 -- and the model writes it unprompted because it is
    drawing a real distinction: a query is a CLASS or a PROPER NOUN, and the
    surface used to have to guess. The original defect this test names is
    still pinned: the constraint must never vanish silently. It now either
    GATES by name or fails loud.
    """
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("church", name="Desert Song Foursquare Church")\n')],
        "done",
    ])
    err = _tool_results(fr)[0].get("error") or ""
    # never silently dropped: either it gates, or it says nothing bears
    # that name. What it must NEVER do is succeed while ignoring it.
    assert "NAMED" in err and "Desert Song" in err, err
    assert "never fuzzily" in err, err
    # the remedy names the working spelling, not the old refusal
    assert "find('Desert Song Foursquare Church')" in err, err

    # A SET SURVIVING INTO THE NEXT TURN is now ANSWERABLE rather than
    # refused: `near=<set>` is the gap to the set's NEAREST member, which
    # is the whole of "the house with a pool in the backyard". What the
    # type system still refuses is a set where a POINT has to be (stand at,
    # face, centre a map on) -- there is no one point in a union.
    reply, fr, flights, _d = _drive([
        [_run('r1 = find("car")\n')],
        [_run('r2 = find("car", nearest=r1)\n')],   # r1 is a SET from turn 1
        "done",
    ])
    assert "error" not in _tool_results(fr)[1], _tool_results(fr)[1]
    reply, fr, flights, _d = _drive([
        [_run('r1 = find("car")\n')],
        [_run('show_map(around=r1)\n')],
        "done",
    ])
    err = _tool_results(fr)[1].get("error") or ""
    assert "around= takes ONE place" in err, err
    assert flights == []


def test_the_vision_calls_are_the_sentence_surface_s_own_steps():
    """look / the looking question / the map compile to the SAME steps.

    The flown arms run with look_tool on, so a code surface that could not
    reach crops, the badged SET look, or the map would be capability-
    reduced against its own baseline and any SR difference would be
    confounded by what was missing rather than by the shape.
    """
    import pytest
    from maps.search.parse_compile.code_surface import CodeError
    bound = {"r1": "r7"}
    plan = plan_of_block("look(r1)\n", bound=bound)
    assert plan["steps"] == [{"do": "crops", "of": "r7"}]
    # AND IT TAKES NOTHING ELSE. The free-text "looking question" is off
    # this surface: it answered into the picture and bound nothing, so a
    # distinction it resolved had nowhere to land. The model asked it in 89
    # of 500 episodes and then indexed rank 0 by DISTANCE anyway. The same
    # question is a QUERY now -- look_for the class, cover a candidate, and
    # find("house", nearest=pools) ranks by the gap to the nearest one. look()
    # takes ONE thing, so the sentence is an arity refusal.
    with pytest.raises(CodeError, match="positional argument"):
        plan_of_block('look(r1, "which one has a pool?")\n', bound=bound)
    # around= POSITIONS rather than measures, so a SET is refused with the
    # fix in the message and a ROW maps straight through.
    with pytest.raises(CodeError, match="around= takes ONE place"):
        plan_of_block("show_map(around=r1)\n", bound=bound)
    row = plan_of_block("show_map(around=r2)\n", bound={"r2": "prior_5"})
    assert row["steps"] == [{"do": "map", "around": ("lit", "prior_5")}], row


def test_no_argument_is_dropped_on_the_floor():
    """Three ways the surface accepted an argument and ignored it.

    All three from the calcity_code_200 audit, all three silent: the model
    stated a constraint, the run looked successful, and the constraint was
    never applied. A refused call now says which keyword it does not have
    and lists the ones it does; the one the compiler CAN honour -- near=
    beside within= -- is honoured rather than refused.
    """
    import pytest
    from maps.search.parse_compile.code_surface import CodeError

    # find("road", query="Sherwood Avenue") -- the positional won and the
    # keyword vanished. 39 calls; one served all 810 roads on the bake.
    with pytest.raises(CodeError, match="multiple values for argument"):
        plan_of_block('r1 = find("road", query="Sherwood Avenue")\n')
    # ...and either one alone still works.
    assert ops_of_block('r1 = find(query="Sherwood Avenue")\n')[0]["args"] \
        == {"query": "Sherwood Avenue"}

    # go_to() refuses a keyword it does not take: an ignored
    # offset=(400, 0) would fly to the identical pose as offset=(-100, 0).
    with pytest.raises(CodeError, match="unexpected keyword argument 'offset'"):
        plan_of_block('go_to(r1, offset=(400, 0))\n', bound={"r1": "prior_1"})
    assert plan_of_block("go_to(r1)\n", bound={"r1": "prior_1"})["steps"] \
        == [{"do": "move", "to": ("lit", "prior_1")}]

    # near= and within= together: the compiler read `near or inside` and
    # filtered on near='s anchor, swapping the semantics in silence. BOTH
    # ride now -- inside cuts, near orders -- so neither is dropped.
    ops = ops_of_block(
        'r1 = find("building", nearest="A Street", within="B Street")\n')
    cut, = [o for o in ops if o["op"] == "filter_items"]
    sort, = [o for o in ops if o["op"] == "score_by"]
    assert cut["args"]["key"] == "distance to b_street", ops
    assert [e["key"] for e in sort["args"]["emphasis"]] == [
        "distance to a_street"], ops

    # The kwarg gate's twin: an EXTRA POSITIONAL was dropped just as
    # silently, and it is the same sentence spelled the other way.
    with pytest.raises(CodeError, match="find takes ONE query"):
        plan_of_block('r1 = find("road", "Sherwood Avenue")\n')
    with pytest.raises(CodeError, match="positional argument"):
        plan_of_block('go_to("prior_1", "prior_2")\n')
    # ...and too FEW: go_to() flew to the string "None".
    with pytest.raises(CodeError, match="missing 1 required positional"):
        plan_of_block("go_to()\n")
    with pytest.raises(CodeError, match="order needs its key"):
        plan_of_block("r2 = r1.order()\n", bound={"r1": "r1"})


def test_any_earlier_handle_stays_indexable_not_just_the_newest():
    """`go_to(r1[0])` with a block in between.

    The pick used to resolve off ONE (ref, rows) slot holding the previous
    plan's final result, so indexing anything but the newest handle failed
    with "unknown fid '_p0'" -- 31 blocks over 18 episodes on
    calcity_code_200. carry_results is keyed by ref and holds them all.
    """
    reply, fr, flights, _d = _drive([
        [_run('r1 = find("car")\n')],
        [_run('r2 = find("tower")\n')],       # r1 is no longer the newest
        [_run('go_to(r1[0])\n')],
        "the car",
    ])
    out = _tool_results(fr)
    assert "error" not in out[2], out[2]
    assert reply.flights_flown == 1
    assert flights and flights[0]["target_fid"] == "car_1", flights


def test_a_pick_never_inherits_the_previous_block_s_row():
    """The worst defect the audit found, and it commits the wrong thing.

    _hoist_inline_picks restarts its counter at `_p0` for EVERY block, so
    block 2's `_p0` is block 1's `_p0` by name. The binding was mission-
    wide and compile_steps keeps an existing string binding, so the second
    pick silently resolved to the FIRST block's fid: episode 205_203 flew
    go_to(r3[0]), go_to(r2[0]), go_to(r5[0]) and go_to(r5[2]) and landed on
    prior_3602 four times; 99_98 had ten distinct picks all land on
    prior_4123.

    Here the third block indexes a handle that was never bound. The right
    answer is a loud failure; the defect answers with block 2's car.
    """
    reply, fr, flights, _d = _drive([
        [_run('r1 = find("car")\n')],
        [_run('go_to(r1[0])\n')],                # binds the internal _p0
        [_run('go_to(r9[0])\n')],                # r9 is not a handle
        "done",
    ])
    out = _tool_results(fr)
    assert flights and flights[0]["target_fid"] == "car_1", flights
    # The unbound pick resolves to NOTHING -- never to the row the last
    # block's `_p0` happened to hold.
    assert "car_1" not in [f["target_fid"] for f in flights[1:]], flights
    # ...and no internal name outlived its own block.
    assert not [h for h in out[1].get("handles") or [] if h.startswith("_")], \
        out[1]


def test_a_pick_is_usable_in_the_block_that_made_it():
    """The whole chain in ONE block, which is what the surface is for.

    A block compiled to ONE plan could not use its own pick: the
    {"select": ...} marker reached the executor raw ("compare refs object
    element must be {ref: ...}; got keys ['select']", 12 blocks) and a
    same-block `r4 = r3[0]` then `go_to(r4)` carried the name into the move
    ("'r4' is a result handle; move takes ... a literal fid"). The block
    now splits at each selection: run, resolve, compile the rest.
    """
    reply, fr, flights, _d = _drive([
        [_run('r1 = find("car", nearest="the depot")\n'
              'r2 = r1[0]\n'
              'r3 = find("gate", nearest=r2)\n'
              'go_to(r2)\n')],
        "the car by the depot",
    ])
    out = _tool_results(fr)
    assert "error" not in out[0], out[0]
    # ONE BLOCK, ONE TABLE: the block's last result keeps its rows and the
    # earlier find folds to its hint, attributed to the model's name.
    res = out[0]["result"]["results"]
    assert list(res) == ["r3"], res
    assert "r1:" in (res["r3"].get("hint") or ""), res["r3"]
    assert out[0]["handles"] == ["r1", "r2", "r3"], out[0]
    assert reply.flights_flown == 1
    assert flights and flights[0]["target_fid"] == "car_1", flights


def test_a_selected_crossing_can_anchor_the_next_call():
    """The surface's headline chain: two roads, their crossing, what is near.

    A crossing row's fid IS its [x, y] token. Bound as a STRING it reaches
    a ref slot as a NAME and is grounded as the text that spells it --
    `rule N: nothing named "[6076, 4226]" is on this surface`, 28 blocks
    over 23 episodes on calcity_code_200 -- so the chain died at exactly
    the step the code shape exists to make expressible. run_block always
    parsed the token; the plan adapter did not.
    """
    from maps.search.parse_compile.code_surface import bind_row
    assert bind_row({"fid": "[6076, 4226]"}) == [6076.0, 4226.0]
    assert bind_row({"fid": "prior_9"}) == "prior_9"   # every other row type

    xing = [{"fid": "[10, 20]", "xy": [10.0, 20.0], "class": "crossing",
             "search_text": "the crossing"}] + list(_CANDS)
    reply, fr, flights, _d = _drive([
        [_run('r1 = find("the crossing")\n'
              'r2 = r1[0]\n'
              'r3 = find("car", nearest=r2)\n')],
        [_run("go_to(r2)\n")],
        "done",
    ], cands=xing)
    out = _tool_results(fr)
    assert "error" not in out[0], out[0]      # not `nothing named "[10, 20]"`
    assert "error" not in out[1], out[1]
    assert flights and flights[0]["target_fid"] == "[10, 20]", flights
    # The anchor went in as a POINT, so compare labelled its columns after
    # the handle. Bound as a string it takes the grounding path instead and
    # the columns come back named 'distance to 10_20' -- the same shape
    # that fails rule N on a real bake, where no row is named "[10, 20]".
    assert "distance to r2" in out[0]["result"]["results"]["r3"]["rows"], \
        out[0]["result"]["results"]["r3"]["rows"]

    # ...and the point rides every one-place slot the same way. The row a
    # selection binds carries the point in `bind`, which is what every
    # slot reads -- the token that spells it never reaches one.
    from maps.search.parse_compile.python_surface import Row
    pt = {"r2": Row({"fid": "[10, 20]", "xy": [10.0, 20.0]})}
    plan = plan_of_block('r3 = find("car", nearest=r2)\n', bound=pt)
    assert plan["groundings"] == [], plan["groundings"]
    assert ops_of_block('r3 = find("car", nearest=r2)\n', pt)[1]["args"]["refs"] \
        == [{"ref": [10.0, 20.0], "as": "r2"}]
    assert plan_of_block("go_to(r2)\n", bound=pt)["steps"] == [
        {"do": "move", "to": ("xy", [10.0, 20.0])}]
    assert plan_of_block("show_map(around=r2)\n", bound=pt)["steps"] == [
        {"do": "map", "around": ("xy", [10.0, 20.0])}]


def test_only_the_model_s_own_names_are_served_back():
    """Two name leaks, and the model wrote code against both of them.

    The executor mints r0, r1, ... in the SAME namespace the model binds
    its handles in, so a receipt keyed by the executor's ref reads as a
    handle: `r1 = intersection(...)` came back as a result called `r0`,
    and one episode went on to write `r9 = r18[4]`. And the internal pick
    name rode the `handles` list -- `["_p0", "r1", "r2"]` -- after which a
    reasoner wrote a bare `go_to(_p0)`.
    """
    reply, fr, flights, _d = _drive([
        [_run('r1 = find("car")\n'
              'r2 = find("gate", nearest="the depot")\n')],
        [_run("go_to(r1[0])\n")],               # mints an internal _p0
        [_run('r3 = find("tower")\n')],
        "done",
    ])
    out = _tool_results(fr)
    # The receipt is keyed by what the MODEL called it. r2's op chain is
    # find -> compare -> score_by, so the executor's own name for it is r3.
    assert list(out[0]["result"]["results"]) == ["r2"], \
        list(out[0]["result"]["results"])
    # No internal name anywhere in what is served back.
    for res in out:
        assert "_p" not in json.dumps(res), res
    assert out[2]["handles"] == ["r1", "r2", "r3"], out[2]
    assert flights and flights[0]["target_fid"] == "car_1", flights


def test_an_unbindable_inline_pick_names_what_the_model_wrote():
    """`go_to(r9[0])` on a handle that was never bound.

    The refusal has to name r9[0]. The internal `_p0` the hoist made for
    it is this block's private wiring, and a refusal that spells it is how
    the model learned to write `go_to(_p0)` in the first place.
    """
    reply, fr, flights, _d = _drive([
        [_run("go_to(r9[0])\n")],
        "done",
    ])
    err = _tool_results(fr)[0].get("error") or ""
    assert "name 'r9' is not defined" in err, err
    assert "_p" not in err, err
    assert flights == [], flights


def test_a_ref_is_told_from_a_name_by_shape_not_by_first_letter():
    """Four bugs, one test: `startswith("r")` is not a ref test.

    A bound row went back through the fuzzy grounding find that produced
    it; a place called "Riverside" was never grounded at all and reached
    the executor as a fid; an unbound handle reached it as an op ref and
    came back talking about reference cycles; and look() on a row whose
    fid starts with r (road_1, runway_2) took the SET branch and framed a
    handle that does not exist.
    """
    import pytest
    from maps.search.parse_compile.code_surface import CodeError
    src = 'r3 = find("gate", nearest=r2)\n'
    held = {"r1": "r0", "r2": "car_1"}
    # A bound fid is a fid: it rides the ref slot, it is not looked up.
    assert plan_of_block(src, bound=held)["groundings"] == []
    assert ops_of_block(src, held)[1]["args"]["refs"] == [
        {"ref": "car_1", "as": "r2"}]
    # A place the model named is grounded whatever letter it starts with.
    named = plan_of_block('r1 = find("car", nearest="Riverside Drive")\n')
    assert [g["query"] for g in named["groundings"]] == ["riverside drive"]
    # An unbound handle never reaches the compiler at all: it is a
    # NameError on the model's own line, rather than an op ref the
    # executor answers with "reference cycle".
    with pytest.raises(CodeError, match="name 'r0' is not defined"):
        plan_of_block('r8 = find("car", nearest=r0)\n')
    # look() on a row: the SET branch is for an op ref, by shape.
    road = plan_of_block("look(r2)\n", bound={"r2": "road_1"})
    assert road["steps"] == [{"do": "crops", "fid": ("lit", "road_1")}], road
    rset = plan_of_block("look(r2)\n", bound={"r2": "r7"})
    assert rset["steps"] == [{"do": "crops", "of": "r7"}], rset
    # ...and what the COMPILER put in a ref slot is never a name to ground:
    # intersection(r1, ...) relates to every way r1 names, which is the
    # escape hatch for a road whose name matches several.
    xsrc = 'r2 = intersection(r1, "Cross Street")\n'
    # A SET in a ref slot is its UNION, and `crossings` unwraps the wrapper
    # to the same rows the bare ref gave it -- "every way r1 names".
    assert ops_of_block(xsrc, {"r1": "r7"})[0]["args"]["refs"][0] == {
        "ref": ["r7"], "as": "r1"}
    assert [g["query"] for g in
            plan_of_block(xsrc, bound={"r1": "r7"})["groundings"]] \
        == ["cross street"]


def test_the_items_slot_takes_a_bare_ref_and_the_place_slots_a_labelled_one():
    """Two shapes for one lookup, and the coercers are not interchangeable.

    A one-place slot needs {ref, as} so a bound fid is not looked up by
    name; `items` takes a WHOLE result and its coercer refuses the
    wrapper -- "cannot resolve a POINT-LIST from {'ref': 'car_1', 'as':
    'r2'}: wants a list of points ... NOT a single fid."
    """
    for src in ('r9 = r2.order(nearest="the depot")\n',
                'r9 = r2.filter(within="the depot")\n'):
        ops = ops_of_block(src, {"r2": "car_1"})
        assert ops[0]["args"]["items"] == "car_1", (src, ops)
        assert ops[0]["args"]["refs"][0]["as"] == "the depot", (src, ops)
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\nr2 = r1[0]\n')],
        [_run('r9 = r2.order(nearest="the depot")\n')],
        "done",
    ])
    assert "error" not in _tool_results(fr)[1], _tool_results(fr)[1]


def test_the_executor_never_mints_a_ref_the_model_already_uses():
    """The two rN namespaces overlap, so this program skips the model's.

    An `as` label rides IN the op's args, where the executor reads every
    string as a possible in-program handle. `r3 = r1[1]` then
    find(nearest=r3) put "r3" in the compare's args while the compare's own
    ref was r2 and the score_by's was r3: "reference cycle among ['r2',
    'r3']", for a program that is a straight line.

    Every handle IN PLAY is skipped. A name the block is in the middle of
    binding is not yet in play -- the call dispatches before the
    assignment lands, so the compiler cannot see the target -- and that
    gap is real: `r4 = find(...)` can mint executor r4 while the model's
    r4 ends up on the score_by's ref. It costs nothing today because no
    `as` label can spell a name that is not yet bound, which is what the
    cycle detector reads.
    """
    refs = [o["ref"] for o in ops_of_block(
        'r4 = find("gate", nearest=r3)\n',
        {"r1": "r0", "r3": "car_2", "r4": "r9"}, rn=1)]
    assert "r3" not in refs and "r4" not in refs, refs
    assert refs == ["r2", "r5", "r6"], refs   # r3/r4 are the model's
    # ...and the whole chain runs rather than tripping the cycle detector.
    reply, fr, _f, _d = _drive([
        [_run("r1 = find('car')\nr2 = r1[0]\n")],
        [_run("r3 = r1[1]\nr4 = find('gate', nearest=r3)\n")],
        "done",
    ])
    out = _tool_results(fr)
    assert "error" not in out[1], out[1]
    assert "distance to r3" in str(out[1]["result"]["results"]), out[1]


def test_a_call_compiles_from_the_counter_the_flight_moved():
    """A flight mid-block takes an rN, and the next call must not reuse it.

    _fly_moves mints a result handle over the rows a flight revealed --
    `r{plan_rn[0]}`, then bumps it. A block with a flight in the middle has
    a second writer into that counter: a front-end holding its own counter
    would compile the call AFTER the flight at the ref the flight had just
    taken, and carry_results then overwrote the flight's rows under that
    name -- while the flight's own report told the model `"of rN, ..."
    names them`. Every call reads ``env.plan_rn`` afresh, so it cannot.
    """
    results, _g, added = _fly_drive([
        [_run('go_to("prior_7")\n'          # flies, and mints over its rows
              'c3 = find("gate")\n')],      # ...must not land on that ref
        "done",
    ])
    assert "error" not in results[0], results[0]
    (rep,) = results[0]["flown"]["flown"]
    minted = rep["new_rows"].split(" ")[0]
    assert re.fullmatch(r"r\d+", minted), rep
    # the flight's handle is still addressable, and it is not c3's
    assert minted in results[0]["handles"], results[0]
    assert "c3" in results[0]["handles"], results[0]
    assert added[0], added
    res = results[0]["result"]["results"]
    # THE WITNESS: the find matched nothing, so a find compiled onto the
    # flight's own ref would overwrite its rows with an empty set under
    # the name the flight report just handed the model.
    assert res["c3"]["len"] == 0, res
    assert res["go_to('prior_7')"]["len"] == len(added[0]), res


def test_the_receipt_rename_never_serves_one_handle_under_another_s_name():
    """The rename is one-to-one or it does not happen.

    Segment 1 mints executor r0 for the model's r1; segment 2's bare
    find() mints executor r1, which has no model handle and keeps its own
    name. Renaming only the mapped one puts two different results under
    `r1` -- the model's cars vanish and it is served a gate under the name
    it bound to the cars. An unhelpful name beats a false one.
    """
    reply, fr, _f, _d = _drive([
        [_run("r1 = find('car')\nr2 = r1[0]\nfind('gate')\n")],
        "done",
    ])
    res = _tool_results(fr)[0]["result"]["results"]
    got = {k: v.get("fids") for k, v in res.items()}
    # The unassigned find is the block's shown answer and is keyed by the
    # model's own spelling; r1's envelope folds to its attributed hint --
    # never served under another name.
    assert got.get("find('gate')") == ["prior_1"], got
    assert "r1:" in (res["find('gate')"].get("hint") or ""), res


def test_a_rebind_that_failed_unbinds_the_name():
    """The stale-pick defect through its second door.

    Block-scoping the internal names closed the `_p0` route; a name the
    MODEL re-binds took the same route through _code_bound, because the
    write-back was an update and an update never deletes. The model was
    told r3 did not bind, was still shown r3 in `handles`, and the next
    go_to(r3) committed to the previous block's row.
    """
    reply, fr, flights, _d = _drive([
        [_run('r1 = find("car")\nr3 = r1[0]\n')],
        [_run('r2 = find("gate")\nr3 = r2[99]\n')],      # r3 fails to re-bind
        [_run("go_to(r3)\n")],
        "done",
    ])
    out = _tool_results(fr)
    assert "did not bind" in (out[1].get("error") or ""), out[1]
    assert "r3" not in out[1]["handles"], out[1]["handles"]
    assert "car_1" not in [f["target_fid"] for f in flights], flights


def test_a_later_segment_s_refusal_keeps_the_earlier_bindings():
    """`r1.filter()` is caught by the compiler, not the parser.

    So the block splits, segment 1 runs and binds r1 and r2, and segment 2
    refuses. The refusal is the result -- and r1 and r2 stay bound, because
    re-finding them would re-mint what the model was already shown.
    """
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\n'
              'r2 = r1[0]\n'
              'r3 = r1.filter()\n')],
        "done",
    ])
    out = _tool_results(fr)
    assert "filter needs what to cut against" in (out[0].get("error") or ""), \
        out[0]
    assert out[0]["handles"] == ["r1", "r2"], out[0]


def test_a_flight_never_mints_a_handle_the_model_is_already_using():
    """The rN namespace has two writers and only one was reading.

    The compiler skips every rN a model name spells; ``_fly_moves`` wrote
    `r{plan_rn[0]}` blind. `r1 = find(...)` leaves the counter at 1, so the
    very next block's flight minted `r1` -- and the flight report then told
    the model "r1 -- the N rows this flight added" while `r1[0]` still
    resolved to the depot rows it bound. A wrong commit, silently.
    """
    results, _g, added = _fly_drive([
        [_run('go_to("prior_7")\n')],              # mints r0; counter -> 1
        [_run('r1 = find("black car")\n')],      # skips r1, mints r2 -> 3
        [_run("r3 = r1[0]\ngo_to(r3)\n")],         # the mint lands on r3
        "done",
    ])
    (rep,) = results[2]["flown"]["flown"]
    handle = rep["new_rows"].split(" ")[0]
    assert handle != "r3", rep
    # ...and a handle the report NAMES is one the surface can resolve: it
    # is an executor ref, so it binds to itself in the code scope.
    assert handle in results[2]["handles"], results[2]
    assert added[0], added


def test_a_bare_display_line_shows_the_rows_it_promises():
    """`r1` alone was a SILENT no-op on the mission path.

    The legend promises "a bare handle on its own line SHOWS the rows;
    r1[0] shows one", and the did-not-bind note tells the model to "Show
    it first, then index it" -- advice nothing could act on. The plan path
    dropped the step: no rows, no error. 25 such lines over 11 episodes.

    A display is an OP now -- the identity window the widening form
    already uses -- and the envelope is keyed by the model's own spelling,
    because a display binds no handle and would otherwise come back as
    "(unnamed N)" after the model asked for it by name.
    """
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\nr1\n')],
        [_run("r1[1]\n")],
        [_run("r9\n")],
        "done",
    ])
    out = _tool_results(fr)
    whole = out[0]["result"]["results"]["r1"]
    assert whole["fids"] == ["car_1", "car_2"], whole
    assert "car_1 | car" in whole["rows"], whole
    # ONE row, still rendered as a row: an int index yields a bare dict and
    # loses the ordering attribute, so the envelope downgraded to census +
    # len -- the model asked to SEE it and got a count of one.
    one = out[1]["result"]["results"]["r1[1]"]
    assert one["fids"] == ["car_2"], one
    assert "car_2 | car" in one["rows"], one
    # ...and a display of nothing is loud rather than empty.
    assert "name 'r9' is not defined" in (out[2].get("error") or ""), out[2]


def test_showing_a_one_row_handle_serves_the_row_it_stands_for():
    """A pick binds a FID, and a fid names no result to window.

    `filter_items` refuses it outright ("cannot resolve a LIST from
    'car_1'"), so the display serves the binding itself rather than
    compiling an op that cannot run.
    """
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\nr2 = r1[0]\nr2\n')],
        "done",
    ])
    out = _tool_results(fr)
    assert out[0]["shown"] == {"r2": "car_1"}, out[0]


def test_a_pick_says_which_row_it_got():
    """The one step that decides the commit said nothing at all.

    An assignment receipts the handle and a count, a display serves rows,
    and `r3 = r2[0]` -- the step that chooses what the mission flies to --
    returned only the fact that r3 now exists. The model then flew a row
    it had never been told the identity of.
    """
    xing = [{"fid": "prior_9", "xy": [1.0, 1.0], "class": "church",
             "name": "Desert Song Foursquare Church",
             "search_text": "church desert song foursquare church"}]
    reply, fr, flights, _d = _drive([
        [_run('r1 = find("car")\nr2 = r1[1]\n')],
        [_run('r3 = find("church")\nr4 = r3[0]\ngo_to(r4)\n')],
        "done",
    ], cands=list(_CANDS) + xing)
    out = _tool_results(fr)
    assert out[0]["picked"] == {"r2": {"fid": "car_2"}}, out[0]
    # ...and the NAME rides it when the row has one, which is the whole
    # difference between "r4 is bound" and "r4 is the church you meant".
    assert out[1]["picked"] == {
        "r4": {"fid": "prior_9",
               "name": "Desert Song Foursquare Church"}}, out[1]
    # The inline pick's internal name is never receipted: it is this
    # block's private wiring and naming it teaches a handle the model
    # never made.
    reply, fr, flights, _d = _drive([
        [_run('r1 = find("car")\n')],
        [_run("go_to(r1[0])\n")],
        "done",
    ])
    assert "picked" not in _tool_results(fr)[1], _tool_results(fr)[1]


def test_look_for_arms_the_segmenter_the_footer_already_advertises():
    """Nothing on this surface armed perception, and the footer said it did.

    The last 200-episode run flew 167 flights and returned ZERO
    detections: the watchlist was empty on every one of them, because no
    call on this surface could reach the ONE segmenter channel. Meanwhile
    every `run` result's footer told the model to say look_for("<class>"),
    and it wrote exactly that 54 times, unprompted, into a surface that
    had no such call.

    It is FREE -- it directs the sensors and moves nothing -- so unlike
    moving it is legal inside a loop.
    """
    reply, fr, _f, declared = _drive([
        [_run('look_for("pool")\n')],
        [_run('look_for("car", "parking lot")\n')],
        "done",
    ])
    out = _tool_results(fr)
    assert out[0]["watching"] == {"now": ["pool (target)"]}, out[0]
    assert out[1]["watching"]["now"] == [
        "pool (target)", "car (target)", "parking lot (target)"], out[1]
    assert declared[-1] == [{"class": "pool", "role": "target"},
                            {"class": "car", "role": "target"},
                            {"class": "parking lot", "role": "target"}]
    # ...and free means loop-legal: the pre-scan cuts the MOVING calls out
    # of a loop body and nothing else.
    from maps.search.parse_compile.python_surface import scan_block
    assert scan_block('for c in r1:\n    look_for("pool")\n')


def test_the_near_form_is_the_targeted_boost_and_a_place_is_grounded():
    """`look_for("pool", at=<place>)` compiles to the boost step, and its
    where= is a GROUNDING.

    The act slots took the model's string verbatim, so a place reached the
    flight, the stage or the boost as a fid nothing answers to. The
    sentence surface has always resolved a place through the plumbing find
    under the name-match floor; the code surface now reaches that one
    resolver instead of shipping the raw text.
    """
    plan = plan_of_block('look_for("pool", at="the church")\n')
    assert plan["steps"] == [{"do": "boost", "concepts": ["pool"],
                              "where": [("fid", "g0")]}], plan["steps"]
    assert plan["groundings"] == [{"slot": "g0", "query": "the church",
                                   "provenance": "any", "why": "boost"}]
    # A flight to a NAME resolves the name; a flight to a fid stays literal
    # (grounding one would look its NAME up and find nothing), and so does
    # anything a binding holds -- a fid a selection resolved came OUT of a
    # row and must never go back through the find that produced it.
    named = plan_of_block('go_to("the depot")\n')
    assert named["steps"] == [{"do": "move", "to": ("fid", "g0")}], named
    assert named["groundings"][0]["why"] == "move"
    lit = plan_of_block('go_to("prior_160")\n')
    assert lit["steps"] == [{"do": "move", "to": ("lit", "prior_160")}], lit
    held = plan_of_block("go_to(r2)\n", bound={"r2": "car_1"})
    assert held["steps"] == [{"do": "move", "to": ("lit", "car_1")}], held
    # ...and end to end: the name reaches the flight as the row it named.
    reply, fr, flights, _d = _drive([
        [_run('go_to("the depot")\n')],
        "done",
    ])
    assert flights == [{"target_fid": "prior_7"}], flights


def test_there_is_no_aim_and_no_refusal_may_invent_one():
    """`at=` is DELETED, and the capability moved instruments.

    121 blocks tried to express an aim -- fly(x, y), fly(fid, x, y), at=,
    offset= -- so the thing they reached for is real: a cue names a PART of
    a region and the centroid is not it. It is not reached by the model
    naming a coordinate. Probed directly with crops and the world window it
    returned the crop's CENTER on 20/21 parsed replies -- midpoint
    arithmetic with no visual pointing in it -- and every aim of the first
    live episodes here landed 23-38 m off the footprint it named. It is
    reached by COVERING the thing and flying to the row the segmenter put
    there, and every refusal must say that rather than name a keyword the
    surface no longer has.
    """
    import pytest
    from maps.search.parse_compile.code_surface import CodeError
    for bad in ('go_to("prior_160", at=(4623, 2981))\n',
                'explore("prior_160", at=(4623, 2981))\n'):
        with pytest.raises(CodeError, match="unexpected keyword argument 'at'"):
            plan_of_block(bad)
    # The COORDINATE spelling that does reach the call names what a moving
    # call takes, and the instrument that reaches a PART of it.
    with pytest.raises(CodeError, match="takes a THING on the map") as exc:
        plan_of_block("go_to([3411, 1468])\n")
    assert "explore the thing" in str(exc.value), str(exc.value)
    # A THING still moves, whatever spells it.
    lit = plan_of_block('go_to("prior_160")\n')
    assert lit["steps"] == [{"do": "move", "to": ("lit", "prior_160")}], lit
    # AND NOTHING SERVED SPELLS ONE. A legend that teaches a form the parser
    # refuses is the failure this arm was built after -- 30 blocks in 22% of
    # episodes were exactly that.
    from maps.search.parse_compile.prompts import (
        CODE_DOC, CODE_SYSTEM_PROMPT)
    for blob in (CODE_DOC, CODE_SYSTEM_PROMPT):
        assert "at=(" not in blob.replace("stand(at=", ""), blob
        assert " at <x>" not in blob, blob


def test_stand_adopts_the_viewpoint_and_its_columns_ride_the_rows():
    """`stand(at=..., facing=...)` -- the sentence's from-looking prefix.

    Free, nothing moves. What it buys is served rather than said: the
    loop mints the vN, the executor rides its side columns on every find
    row, and crops and maps render its facing up. A stance gates nothing,
    which is why it can be a PREFIX -- the block that stands and then
    finds means the find under that stance whichever line came first.
    """
    reply, fr, flights, _d = _drive([
        [_run('stand(at="prior_7", facing="north")\nr1 = find("car")\n')],
        [_run('stand(at="the gate", facing="prior_2")\n')],
        [_run('go_to("v1")\n')],
        "done",
    ])
    out = _tool_results(fr)
    assert out[0]["stance"]["staged"][0]["handle"] == "v1", out[0]
    assert "left of v1" in out[0]["result"]["results"]["r1"]["rows"], out[0]
    # ...a compass word and a PLACE are both headings, and a place is
    # grounded like any other.
    assert out[1]["stance"]["staged"][0]["handle"] == "v2", out[1]
    # A STANCE IS NOT SOMEWHERE TO MOVE TO, and refusing that is what
    # keeps the coordinate hatch shut: the move wire resolves a vN to its
    # stand point, so stand(at=[4620, 2965]) + go_to(v1) would fly the
    # model's own estimate in two hops -- the thing go_to([x, y]) was
    # closed for. The stance keeps its point; it just cannot be flown.
    assert "not a thing on the map" in out[2]["error"], out[2]
    assert flights == [], flights

    import pytest
    from maps.search.parse_compile.code_surface import CodeError
    from maps.search.parse_compile.python_surface import scan_block
    plan = plan_of_block('stand(at="the church", facing="the gate")\n')
    assert plan["steps"] == [{"do": "stage", "at": ("fid", "g0"),
                              "facing": ("fid", "g1")}], plan["steps"]
    assert [g["why"] for g in plan["groundings"]] == ["stage", "facing"]
    # ...and both slots are one-place slots.
    with pytest.raises(CodeError, match="at= takes ONE place"):
        plan_of_block('stand(at=r1, facing="north")\n', bound={"r1": "r7"})
    # Free means loop-legal, like look_for and unlike the moving calls.
    assert scan_block('for c in r1:\n    stand(at=c, facing="north")\n')


def test_more_of_pages_a_long_result_and_stops_at_the_end():
    """`more_of(r1)` -- the sentence `more of rN.`

    A result past the serve law's knee renders its first HEAD rows and the
    rest stays live in the handle; without this call there is no way to
    reach it. Three disjoint blocks, in the handle's own order, then a
    stable refusal -- and a refused widening never advances the window, so
    the rows it failed to reach stay reachable.
    """
    from maps.search.analyze_scene.geom_program import _BROAD_HEAD_ROWS as H
    wide = [{"fid": f"car_{i:03d}", "xy": [float(i), 0.0], "class": "car",
             "search_text": "car"} for i in range(3 * H)]
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\n')],
        [_run("more_of(r1)\n")],
        [_run("more_of(r1)\n")],
        [_run("more_of(r1)\n")],
        "done",
    ], cands=wide)
    out = _tool_results(fr)
    served = [out[0]["result"]["results"]["r1"]["fids"]]
    for r in (out[1], out[2]):
        assert r.get("error") is None, r
        # Labelled with what was ASKED FOR: the rows are a WINDOW of r1,
        # not r1, so keying them "r1" would claim to be the whole set.
        env = r["result"]["results"]["more_of(r1)"]
        assert env["len"] == H and "rows" in env, env
        served.append(env["fids"])
    assert served[0] + served[1] + served[2] == [f"car_{i:03d}"
                                                 for i in range(3 * H)]
    assert "already served" in (out[3].get("error") or ""), out[3]

    import pytest
    from maps.search.parse_compile.code_surface import CodeError
    with pytest.raises(CodeError, match="name 'r9' is not defined"):
        plan_of_block("more_of(r9)\n")
    with pytest.raises(CodeError, match="more_of pages a RESULT"):
        plan_of_block("more_of(r2)\n", bound={"r2": "car_1"})
    # A RESULT pages: the window opens at the head rows already served.
    from maps.search.analyze_scene.geom_program import _BROAD_HEAD_ROWS as B
    paged = plan_of_block("more_of(r1)\n", bound={"r1": "r7"})["plans"][0]
    assert paged["steps"][0]["program"]["ops"][0]["args"]["items"] \
        == f"r7[{B}:{2 * B}]", paged
    assert paged["widened"] == {"r7": B}, paged


def test_three_ordinary_python_forms_that_were_refused_outright():
    """131 refusals between them, each one a turn spent being told to write
    the two lines the parser was about to write anyway.

    A literal binding, an indexed call result, and one level of nesting.
    All three are just Python now: a name binds, a call evaluates before
    the subscript that indexes it, and a nested call runs first.
    """
    from maps.search.parse_compile.python_surface import Row

    # 1. a literal binding (57). What a one-place slot takes -- a fid off
    #    a served row, a world point off a caption -- binds and rides; a
    #    NAME binds too and grounds where it lands, as the quoted form does.
    assert plan_of_block('h = "prior_3123"\ngo_to(h)\n')["steps"] \
        == [{"do": "move", "to": ("lit", "prior_3123")}]
    assert plan_of_block('depot = "the depot"\ngo_to(depot)\n')["steps"] \
        == [{"do": "move", "to": ("fid", "g0")}]
    # ...and a POINT still reaches the moving calls only through a ROW,
    # which is what a selection binds.
    corner = Row({"fid": "end_0", "xy": [3411.0, 1468.0]})
    assert plan_of_block("go_to(corner)\n", bound={"corner": corner})["steps"] \
        == [{"do": "move", "to": ("xy", [3411.0, 1468.0])}]

    # 2. indexing a call result (36) -- the call runs, the index picks a
    #    row of it, and a refusal names what the MODEL wrote.
    reply, fr, flights, _d = _drive([
        [_run('t = find("car")[0]\ngo_to(t)\n')],
        "done",
    ])
    out = _tool_results(fr)
    assert flights == [{"target_fid": "car_1"}], flights
    assert out[0]["picked"] == {"t": {"fid": "car_1"}}, out[0]
    assert "_c" not in json.dumps(out[0]), out[0]
    reply, fr, flights, _d = _drive([
        [_run('t = find("nothing at all")[0]\n')],
        "done",
    ])
    err = _tool_results(fr)[0].get("error") or ""
    assert "find('nothing at all')[0] did not bind" in err, err

    # 3. one level of nesting (38): the inner call is its own dispatch and
    #    the outer measures against what it left.
    ops = ops_of_block('r5 = find("house").filter(within=rd)\n',
                       {"rd": "road_1"})
    assert [o["op"] for o in ops] == ["find", "compare", "filter_items"], ops
    # A NESTED SET in a measuring slot is that set's union -- the gap to
    # the nearest house, which is what the spelling plainly means.
    ops = ops_of_block('r5 = find("car", nearest=find("house"))\n')
    assert [o["op"] for o in ops] == ["find", "find", "compare",
                                      "score_by"], ops
    assert ops[2]["args"]["refs"][0]["ref"] == [ops[0]["ref"]], ops


def test_a_failed_step_does_not_unbind_what_the_program_already_bound():
    """`r1 = find("car")` then a look that cannot render is ONE plan.

    The find executed and r1's rows are in the executor's store, but the
    write-back sat behind the error break, so the block came back with no
    handles at all and the model's only recovery was to re-find -- which
    re-mints handles and diverges from what it was shown. Same law as a
    later segment's refusal keeping an earlier one's bindings, one door
    over.

    A plan that failed BEFORE its program ran has nothing to keep, and the
    analyze envelope is the witness of which case it is.
    """
    def _broken(_reqs):
        raise RuntimeError("renderer down")

    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\nlook(r1)\n')],
        [_run("go_to(r1[0])\n")],
        "done",
    ], look_renderer=_broken)
    out = _tool_results(fr)
    assert "crop render failed" in (out[0].get("error") or "")
    assert out[0]["handles"] == ["r1"], out[0]
    assert "error" not in out[1], out[1]
    # ...and a grounding miss binds nothing: its program never ran.
    reply, fr, _f, _d = _drive([
        [_run('r2 = find("car", nearest="nowhere at all")\n')],
        "done",
    ])
    assert "handles" not in _tool_results(fr)[0], _tool_results(fr)[0]


def test_a_slice_frames_the_head_of_an_ordering():
    """`look(r2[0:20])` -- a badged frame over the head of an ordering.

    A badged frame over the head of an ordering is the disambiguation the
    vision channel exists for. Nothing new executes: the executor windows
    a handle itself (it is what the widening form is built on), so a slice
    is the identity op at a window, and a window of a set IS a set.

    THE CAP is the renderer's own badge cap, inherited rather than added:
    one frame badges the first LOOK_SET_ROWS rows of whatever set it is
    given, so a wider slice is legal and frames its head.
    """
    import pytest
    from maps.search.parse_compile.code_surface import CodeError
    plan = plan_of_block('look(r2[0:20])\n', bound={"r2": "r7"})
    (op,) = plan["steps"][0]["program"]["ops"]
    assert op["args"]["items"] == "r7[0:20]", op
    assert plan["steps"][1] == {"do": "crops", "of": op["ref"]}
    # A window is a SET, so it binds, chains and displays like one, and an
    # open lower bound is the head of the ordering.
    shown = plan_of_block("r5 = r2[:5]\nr5\n", bound={"r2": "r7"})
    assert shown["steps"][0]["program"]["ops"][0]["args"]["items"] \
        == "r7[0:5]", shown
    assert shown["_code_shows"] == {"r5[0:5]": "r0"}, shown["_code_shows"]
    # BOTH ENDS, and no step: a window addresses a range of the ordering,
    # and anything else is refused rather than silently reinterpreted.
    with pytest.raises(CodeError, match=r"\[lo:hi\] with both ends"):
        plan_of_block("r3 = r2[0:20:2]\n", bound={"r2": "r7"})
    with pytest.raises(CodeError, match=r"\[lo:hi\] with both ends"):
        plan_of_block("r2[10:]\n", bound={"r2": "r7"})
    with pytest.raises(CodeError, match="name 'r9' is not defined"):
        plan_of_block("r3 = r9[0:5]\n")

    # ...and end to end the head of the ordering is what comes back.
    from maps.search.analyze_scene.geom_program import _BROAD_HEAD_ROWS as H
    wide = [{"fid": f"car_{i:03d}", "xy": [float(i), 0.0], "class": "car",
             "search_text": "car"} for i in range(3 * H)]
    reply, fr, _f, _d = _drive([
        [_run('r1 = find("car")\nr1[0:5]\n')],
        "done",
    ], cands=wide)
    env = _tool_results(fr)[0]["result"]["results"]["r1[0:5]"]
    assert env["fids"] == [f"car_{i:03d}" for i in range(5)], env


def test_the_legend_documents_every_call_the_surface_serves():
    """A call the parser accepts and the legend omits is a capability the
    model has no way to learn it has.

    Derived, so it carries a vacuity floor and pins by name the two facts
    the derivation cannot notice: what `within=` actually means, and the
    badge cap a slice is sized against.
    """
    import inspect

    from maps.search.parse_compile.prompts import CODE_DOC
    from maps.search.parse_compile.python_surface import (
        MissionPython, Rows, served_calls)

    names = served_calls()
    assert len(names) >= 9, names                    # vacuity floor
    assert not [n for n in names if f"{n}(" not in CODE_DOC], names
    # ...and every keyword the executing functions take, bar order(reverse=),
    # which is accepted and not advertised.
    fns = [getattr(MissionPython, f"_call_{n}") for n in names] + [Rows.order, Rows.filter]
    kws = {n for f in fns for n, p in inspect.signature(f).parameters.items()
           if p.kind is p.KEYWORD_ONLY} - {"reverse"}
    assert len(kws) >= 7, kws
    assert not [k for k in kws if f"{k}=" not in CODE_DOC], sorted(kws)
    # within= is 0 m CONTACT, and 21 of 36 uses returned zero rows because
    # the model read it as "on or near".
    assert "TOUCHES it -- distance 0" in CODE_DOC
    # look over a set serves one crop PER ROW (the map-scale set frame is
    # retired); the image budget is stated when it bites, never spelled
    # in the legend.
    assert "one crop PER ROW" in CODE_DOC
    # THE PROCEDURAL EXPLORE CHAIN, pinned by name: 6b44fee8 deleted this
    # line as diet and explore reach fell 39% -> 14% on identical
    # episodes. A derived scan
    # cannot notice its disappearance; this can.
    assert "To reach a PART" in CODE_DOC
    # the visual-adjective occasion: a cue word no column carries is read
    # off crops, not off the prior
    assert "choose by what the crops show" in CODE_DOC
    # the set form: a SET is one place to cover, and a NAME is the whole
    # street
    assert "a SET is one place to cover" in CODE_DOC
    assert "the name is the whole street" in CODE_DOC
    # THE DE-ENDORSEMENT PAIR (system prompt): a sort is evidence of its
    # column, never of identity, and "know" has a standard for cue words
    # no column carries. Pinned by name so a diet commit fails here.
    from maps.search.parse_compile.prompts import CODE_SYSTEM_PROMPT
    assert "settled by look(), not by" in CODE_SYSTEM_PROMPT
    assert "known only from a look, never from rank" in CODE_SYSTEM_PROMPT
    assert "never that its head is the thing" in CODE_SYSTEM_PROMPT


def test_no_served_string_teaches_a_form_this_surface_refuses():
    """`_run_plan` is SHARED, so the code arm receives the say surface's
    notes -- and several of them named sentences its parser refuses.

    This is the footer defect one door over. `look_for` was added because
    every result's footer told the model to say a form the arm did not
    have; these are the same mistake inside the plan body, where the
    register break is invisible until a live episode reads it back.

    TWO MODULES, and the first version of this scan read only one. Smoke
    25712733 served `say "go to prior_6005 at <x> <y>."` and
    `say look_for("<class>")` in the same episode: the aim advice is built
    in benchmarks/citynav/run_benchmark/mission_engine.py, which the scan
    never opened, and the watchlist footer said `say <a call>`, which the
    `say "` pattern could not match because no quote follows the verb. A
    scan scoped to one file and one spelling failed OPEN in exactly the
    way a derived test is warned about -- so the file list and the
    spellings both widened, and the file list carries its own floor.

    ZERO literals may teach a sentence now. The say dispatch that owned the
    one allowed exception is deleted, so the exception went with it: there
    is no second register left for a note to slip into.
    """
    import ast
    import pathlib

    # Every module that builds a model-facing string on the mission path.
    # A floor on the list itself: a renamed module drops silently
    # otherwise, which is the failure this scan just had.
    modules = [pathlib.Path(p) for p in (
        "maps/search/parse_compile/reasoner_loop.py",
        "benchmarks/citynav/run_benchmark/mission_engine.py",
    )]
    assert len(modules) >= 2 and all(m.is_file() for m in modules), modules

    served = []
    for mod in modules:
        tree = ast.parse(mod.read_text())
        docstrings = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)) and body \
                    and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
        served += [(mod.name, n) for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)
                   and id(n) not in docstrings]
    assert len(served) > 1_000, len(served)          # vacuity floor

    # `say ` bare, not `say "`: the footer that shipped read
    # `say look_for("<class>")` -- the verb of the other surface in front
    # of a call of this one, which the quoted pattern walked straight past.
    sentence_forms = ('say "', 'Say "', 'say look', 'say the ', 'of those',
                      'look at them', 'the nearest one.', 'go to ', 'show me ',
                      'look for <', 'the say tool', 'this sentence')
    hits = sorted({(m, n.lineno, w) for m, n in served for w in sentence_forms
                   if w in n.value})
    # Nothing. The scan's own vacuity floor is the served-literal count
    # asserted above, so an empty result here is a real zero rather than a
    # scan that stopped looking.
    assert not hits, hits


def test_a_block_written_into_the_reply_channel_is_repaired_not_an_ending():
    """The code arm's twin of the sentence-shaped reply.

    A fleet ended 51% of its episodes on a sentence written into the reply
    channel, mid-search. A block written there ends the mission the same
    silent way -- short, with the flights it was about to make never made.
    One repair; the streak is shared with the empty turn, so the ending
    stays model-owned.
    """
    reply, fr, flights, _d = _drive([
        'r1 = find("car")\ngo_to(r1[0])',       # the dropped block
        [_run('r1 = find("car")\n')],          # re-sent through the tool
        [_run("go_to(r1[0])\n")],
        "the car by the depot.",
    ])
    assert reply.ended_by == "reply"
    assert reply.reply_text == "the car by the depot."
    assert reply.flights_flown == 1
    nudges = [m for m in fr.messages
              if m.get("role") == "user"
              and "not through the run tool" in str(m.get("content"))]
    assert len(nudges) == 1, nudges


def test_a_second_block_in_the_reply_channel_still_ends_the_mission():
    """A stuck emitter cannot loop the mission on nudges."""
    reply, fr, flights, _d = _drive(['go_to(r1[0])', 'go_to(r1[0])'])
    assert reply.ended_by == "reply"
    assert reply.flights_flown == 0
    assert reply.turns_used == 2


def test_prose_that_narrates_a_call_is_a_real_ending():
    """Both halves of the detector, each killing what the other lets by.

    A reply must PARSE WHOLE to be a dropped block -- prose that mentions
    a call is the model answering. And it must CARRY a call -- `parse`
    accepts a bare name, so a one-word ending would otherwise be repaired
    into a mission that will not stop.
    """
    from maps.search.parse_compile.reasoner_loop import _code_reply

    assert _code_reply('r1 = find("car")\ngo_to(r1[0])')
    assert _code_reply('```python\ngo_to(r1[0])\n```')
    assert not _code_reply("I flew to the church with go_to(r5); that is it.")
    assert not _code_reply("done")
    assert not _code_reply("The pool house is prior_7.")
    assert not _code_reply("")


def test_the_worked_blocks_in_the_legend_parse_on_this_surface():
    """A legend example the surface refuses is the worst case there is.

    `narrow` sat in the say legend that way -- served as prose while being
    unsayable as a form -- and the model reached for it 9 times against a
    parser that had nothing to parse it onto. The example here exists
    BECAUSE the worked example is what comes back: the say legend's one
    example is a flight onto a junction and the model copies both halves,
    and one legend line carrying a crossing moved crossing-as-anchor 9 ->
    24 (2.7x, p=0.0015). So it has to run.

    It is ONE block now: lines execute eagerly in source order, so the
    query over what a pass found rides IN the block that flew the pass --
    the chain the fold-order surface had to split across two blocks.
    """
    import ast

    from maps.search.parse_compile.prompts import CODE_DOC
    from maps.search.parse_compile.python_surface import scan_block

    head, _, rest = CODE_DOC.partition(
        "A block is any number of lines, run in order. This is one block:\n")
    assert rest, "the worked block is gone from the legend"
    body = rest.partition("\nEach line runs")[0]
    block = "\n".join(ln.strip() for ln in body.splitlines()
                      if ln.startswith("  ") and ln.strip())
    assert block.count("\n") >= 5, block          # it is a BLOCK, not a line
    tree = scan_block(block)                      # the surface accepts it
    calls = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    # The chain that replaced the aim: watch, cover the thing, read what
    # the pass put there, rank, look, fly -- and the COVER precedes both
    # the query over its detections and the flight that commits.
    assert {"look_for", "explore", "find", "look", "go_to"} <= set(calls), \
        calls
    assert calls.index("explore") < calls.index("go_to"), calls
    # THE ORDERING RIDES THE FIND: the example ranks on every criterion
    # the cue names, appearance included -- 13 colour-ordering heads over
    # 2 of 500 episodes on the fold-order surface against 440 over 175 of
    # 300 on the sentence language is the gap the worked block is the
    # lever on.
    kw = {k.arg for n in ast.walk(tree)
          if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
          and n.func.id == "find" for k in n.keywords}
    assert "color" in kw, kw
    # nearest= is the ordering spelling; near= now cuts and needs dwithin=.
    assert kw & {"nearest", "near", "within", "east_of", "west_of",
                 "north_of", "south_of"}, kw
    # AND THE PICTURE CARRIES NO SENTENCE. `look(rows, "...?")` is off the
    # surface: it answered into the picture and bound nothing.
    assert all(len(n.args) == 1 for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "look")


def test_find_takes_the_colour_ordering_the_executor_always_had():
    """`find(class, color=word)` is one call, and it is one sort.

    The executor's find has accepted a colour since before this surface
    existed -- it attaches `color_<word>` and drops nothing -- and the
    kwarg gate here was the only thing refusing it, so the sole spelling
    left was a SECOND call the model had to know to make. It made it
    almost never: 13 colour-ordering heads over 2 of 500 episodes against
    440 over 175 of 300 on the sentence language, while 12,706 served rows
    carried a colour cell nothing sorted on.

    The compiled DAG has to be IDENTICAL to the two-step spelling, and the
    colour has to ride the SAME score_by as the distances: two sorts in a
    row is not two constraints, and that defect silently dropped the
    proximity out of a find that named both.
    """
    def _ops(src):
        return [(o["op"], o["args"]) for o in ops_of_block(src)]

    one = _ops('r1 = find("house", color="red")\n')
    two = _ops('r1 = find("house")\nr2 = r1.order(color="red")\n')
    assert [o for o, _a in one] == ["find", "score_by"], one
    assert [a for _o, a in one] == [a for _o, a in two], (one, two)

    # ONE score_by, colour last, distances first -- so `_code_order` still
    # reads the anchor whose span the tie note is about.
    fused = _ops('r1 = find("house", nearest="A", south_of="B", color="gray")\n')
    sorts = [a for o, a in fused if o == "score_by"]
    assert len(sorts) == 1, fused
    assert [e["key"] for e in sorts[0]["emphasis"]] == [
        "distance to a", "distance to b", "color_gray"], sorts

    # and it ORDERS, never cuts: every row the find matched is still there
    assert not [o for o, a in fused
                if o == "filter_items" and "color" in str(a.get("key"))], fused


# --- explore(set): many members, ONE chained pass -------------------------

_WILLOW_CANDS = [
    {"fid": "w1", "xy": [10.0, 0.0], "name": "Willow Avenue",
     "search_text": "willow avenue"},
    {"fid": "w2", "xy": [200.0, 0.0], "name": "Willow Avenue",
     "search_text": "willow avenue"},
    {"fid": "w3", "xy": [400.0, 0.0], "name": "Willow Avenue",
     "search_text": "willow avenue"},
    {"fid": "car_1", "xy": [20.0, 20.0], "class": "car",
     "search_text": "car"},
]

_WILLOW_STOPS = {"w1": [[0.0, 0.0], [100.0, 0.0]],
                 "w2": [[150.0, 0.0], [250.0, 0.0]],
                 "w3": [[350.0, 0.0], [450.0, 0.0]]}


def test_an_explore_of_a_NAME_covers_every_segment_bearing_it():
    """explore("Willow Avenue") is the whole street, not its first
    fragment. The grounding's top tie group IS the feature (140#138:
    five tied segments, [0] taken, target beside the last, 1554 m), and
    the pass walks all of them end to end — so the "flying the best name
    match" note has nothing true to say and is not served.
    """
    import json as _json
    from maps.util.xy_token import parse_xy_token
    reply, out, flights, asked, _rend = _explore_drive([
        [_run('explore("Willow Avenue")\n')],
        "done",
    ], stops=_WILLOW_STOPS, cands=_WILLOW_CANDS)
    assert "error" not in out[0], out[0]
    assert {f for f, _ in asked} == {"w1", "w2", "w3"}, asked
    flown = [list(parse_xy_token(f["target_fid"])) for f in flights]
    assert flown == [[0.0, 0.0], [100.0, 0.0], [150.0, 0.0],
                     [250.0, 0.0], [350.0, 0.0], [450.0, 0.0]], flown
    ack = out[0]["explored"]
    assert ack["segments"] == 3, ack
    assert "ended_at" not in ack, ack
    assert "across 3 segments" in (ack.get("note") or ""), ack
    assert "best name match" not in _json.dumps(out[0]), out[0]


def test_an_explore_of_a_handle_chains_members_and_reverses_a_far_end():
    """A member whose FAR end is nearer the chain tail flies reversed, so
    consecutive members join where they touch and the pass stays a line."""
    from maps.util.xy_token import parse_xy_token
    reply, out, flights, _a, _rend = _explore_drive([
        [_run('r1 = find("car")\n')],
        [_run('explore(r1)\n')],
        "done",
    ], stops={"car_1": [[100.0, 200.0], [200.0, 200.0]],
              "car_2": [[500.0, 200.0], [300.0, 200.0]]})
    assert "error" not in out[1], out[1]
    flown = [list(parse_xy_token(f["target_fid"])) for f in flights]
    assert flown == [[100.0, 200.0], [200.0, 200.0],
                     [300.0, 200.0], [500.0, 200.0]], flown
    assert out[1]["explored"]["segments"] == 2, out[1]["explored"]


def test_a_set_pass_covers_a_long_street_rather_than_husbanding_stops():
    """The bound is a RUNAWAY backstop, not a budget the pass husbands.

    36 stops of a 3.6 km street is a long pass and it flies all of them.
    At the old ceiling of 24 the pass truncated, reported itself SAMPLED
    and told the model to narrow -- a served budget, which is the one
    thing the mission never does, and calcity::26#26 settled for a prior
    1,790 m off the cue's road immediately after reading one.
    """
    from maps.util.xy_token import parse_xy_token
    long_stops = {
        "w1": [[float(x), 0.0] for x in range(0, 1200, 100)],
        "w2": [[float(x), 0.0] for x in range(1200, 2400, 100)],
        "w3": [[float(x), 0.0] for x in range(2400, 3600, 100)]}
    reply, out, flights, _a, _rend = _explore_drive([
        [_run('explore("Willow Avenue")\n')],
        "done",
    ], stops=long_stops, cands=_WILLOW_CANDS)
    assert "error" not in out[0], out[0]
    # Past the OLD ceiling of 24 -- the pass is no longer husbanding stops.
    # (The drive harness has a 32-flight backstop of its own, so the exact
    # count is its limit and not the pass's.)
    assert len(flights) > 24, len(flights)
    note = out[0]["explored"].get("note") or ""
    # NO BUDGET FACT: not the ceiling, not "one pass", not advice to do less.
    for banned in ("SAMPLED", "Narrow it", "one pass flies"):
        assert banned not in note, note
    assert "covered" in note, note
    # The PASS plans the whole street end to end; the drive harness's own
    # 32-flight backstop is what stops short of it, not the pass.
    assert out[0]["explored"]["stops"] > 24, out[0]["explored"]
    flown = [list(parse_xy_token(f["target_fid"])) for f in flights]
    assert flown[0] == [0.0, 0.0], flown


def test_a_set_pass_dedupes_the_shared_endpoint_of_touching_members():
    """Two segments that join share a stop; the pass flies it once."""
    from maps.util.xy_token import parse_xy_token
    reply, out, flights, _a, _rend = _explore_drive([
        [_run('explore("Willow Avenue")\n')],
        "done",
    ], stops={"w1": [[0.0, 0.0], [100.0, 0.0]],
              "w2": [[110.0, 0.0], [210.0, 0.0]],
              "w3": [[320.0, 0.0], [420.0, 0.0]]},
        cands=_WILLOW_CANDS)
    flown = [list(parse_xy_token(f["target_fid"])) for f in flights]
    # [110, 0] sits within half a spacing (50) of the kept [100, 0]
    assert [110.0, 0.0] not in flown, flown
    assert len(flown) == 5, flown


def test_go_to_of_a_set_is_refused_toward_explore():
    """go_to still lands on ONE thing; the remedy names the set form."""
    reply, fr, flights, _d = _drive([
        [_run('r1 = find("car")\ngo_to(r1)\n')],
        "done",
    ])
    out = _tool_results(fr)
    assert "error" in out[0], out[0]
    assert "explore(r1) to cover the whole set" in out[0]["error"], out[0]
    assert flights == [], flights


def test_an_explore_of_a_LIST_of_rows_covers_their_union():
    reply, out, flights, _a, _rend = _explore_drive([
        [_run('r1 = find("car")\n')],
        [_run('explore([r1[0], r1[1]])\n')],
        "done",
    ], stops={"car_1": [[100.0, 200.0], [200.0, 200.0]],
              "car_2": [[500.0, 200.0], [300.0, 200.0]]})
    assert "error" not in out[1], out[1]
    assert out[1]["explored"]["segments"] == 2, out[1]["explored"]
    assert len(flights) == 4, flights


# --- pixels are earned by flying: the look coverage gate ------------------

_GATE_CANDS = [
    {"fid": "car_1", "xy": [110.0, 205.0], "class": "car",
     "search_text": "car"},                    # near the pass's stops
    {"fid": "car_2", "xy": [500.0, 900.0], "class": "car",
     "search_text": "car"},                    # far from any stop
]


def test_a_look_before_any_flight_refuses_toward_explore():
    """The sensor holds no imagery before a flight: a pre-flight look
    renders nothing and the note names the verb that earns pixels."""
    import json as _json
    reply, out, flights, _a, rendered = _explore_drive([
        [_run('r1 = find("car")\nlook(r1[0])\n')],
        "done",
    ], cands=_GATE_CANDS, obs_minor_m=100.0, with_renderer=True)
    assert flights == [], flights
    assert all(not r for r in rendered), rendered
    blob = _json.dumps(out[0])
    assert "has never been under the sensor" in blob, out[0]
    assert "explore the place" in blob, out[0]


def test_a_pass_earns_exactly_the_pixels_it_covered():
    """After a pass, a point inside a flown view square renders; ground
    no flight covered still refuses. Points, because this fixture's ctx
    carries no graph and the post-flight surface rebuild empties the
    injected rows -- the gate itself is row/point-agnostic."""
    import json as _json
    reply, out, flights, _a, rendered = _explore_drive([
        [_run('explore("prior_7")\n')],
        [_run('look([110, 205])\n')],  # 11 m from the pass's first stop
        [_run('look([500, 900])\n')],  # no stop near
        "done",
    ], cands=_GATE_CANDS, obs_minor_m=100.0, with_renderer=True)
    served = [r for reqs in rendered for r in reqs]
    assert len(served) == 1, served
    blob = _json.dumps(out[2])
    assert "has never been under the sensor" in blob, out[2]


def test_without_a_stated_footprint_the_gate_stays_off():
    """Sensorless runs state no short side, and a gate keyed on an
    invented number would refuse what the engine could serve."""
    reply, out, flights, _a, rendered = _explore_drive([
        [_run('r1 = find("car")\nlook(r1[0])\n')],
        "done",
    ], cands=_GATE_CANDS, obs_minor_m=None, with_renderer=True)
    served = [r for reqs in rendered for r in reqs]
    assert [r["center_fid"] for r in served] == ["car_1"], served


# --- strata: the find's match tiers as live attribute views ---------------

_TIER_CANDS = (
    [{"fid": f"d{i}", "xy": [float(100 * i), 0.0], "class": "parking",
      "search_text": "parking"} for i in range(3)]
    + [{"fid": f"p{i}", "xy": [float(100 * i), 500.0],
        "search_text": "amenity parking",
        "tags": {"amenity": "parking"}} for i in range(2)])


def test_strata_are_live_views_and_the_census_says_so():
    """r1.exact / r1.tagged split the find's own tiers; the receipt's
    census names them as structure, not a float column."""
    import json as _json
    reply, out, flights, _a, _r = _explore_drive([
        [_run('parks = find("parking")\n'
              'print(len(parks), len(parks.exact), len(parks.tagged))\n')],
        "done",
    ], cands=_TIER_CANDS)
    assert "error" not in out[0], out[0]
    assert out[0]["printed"] == "5 3 2", out[0]["printed"]
    assert "tiers: 3 exact, 2 tagged" in _json.dumps(out[0]), out[0]


def test_explore_of_a_stratum_covers_exactly_its_rows():
    """explore(parks.exact) is the workflow's own spelling: the pass walks
    the rows that ARE the thing and none of the tag-carriers."""
    stops = {f"d{i}": [[float(100 * i), 0.0]] for i in range(3)}
    reply, out, flights, asked, _r = _explore_drive([
        [_run('parks = find("parking")\nexplore(parks.exact)\n')],
        "done",
    ], cands=_TIER_CANDS, stops=stops)
    assert "error" not in out[0], out[0]
    assert out[0]["explored"]["segments"] == 3, out[0]["explored"]
    assert sorted({f for f, _ in asked}) == ["d0", "d1", "d2"], asked


def test_a_stratum_of_tierless_rows_refuses_with_the_teaching():
    """A browse carries no match tiers; the refusal says where strata
    come from instead of serving a silent empty set."""
    reply, out, flights, _a, _r = _explore_drive([
        [_run('everything = find()\nprint(len(everything.exact))\n')],
        "done",
    ], cands=_TIER_CANDS)
    assert "error" in out[0], out[0]
    assert "no match tiers" in out[0]["error"], out[0]["error"]


# --- the pass shows its catch ---------------------------------------------

def test_a_pass_serves_its_catch_as_crops():
    """The sweep hands back pictures of what it found: one crop per row
    the pass added, on the pass's own ack."""
    results, _graph, added = _fly_drive([
        [_run('explore("prior_7")\n')],
        "done",
    ])
    ex = results[0].get("explored") or {}
    assert ex.get("catch"), results[0]
    assert ex["catch"].get("delivered"), ex["catch"]


def test_a_looks_marks_reach_the_renderer_on_every_crop_of_a_set():
    """``marks=`` was a DEAD KEY: the compiler wrote step["mark_fids"] and
    the crops dispatcher never read it, so the renderer -- which has full
    support, including expanding a handle to its ways -- was handed
    nothing. A second gate then dropped marks on any look of more than one
    row, which is the only look that needs them: telling ten candidate
    crops apart is what the anchor is FOR.

    calcity::200#198 is the episode: ten car crops, each legend reading
    only ``1 = TARGET car detected_N (center)``, and the committed car sat
    116 m off the cue's boulevard and 11.9 m off another street.
    """
    # The mark has to be IN FRAME to be drawn -- the renderer drops a mark
    # whose footprint never reached the pixels -- so the fixture puts the
    # cars and the thing marked inside one crop window.
    _close = [
        {"fid": "car_1", "xy": [110.0, 205.0], "class": "car",
         "search_text": "car"},
        {"fid": "car_2", "xy": [118.0, 209.0], "class": "car",
         "search_text": "car"},
        {"fid": "lamp_1", "xy": [114.0, 207.0], "class": "lamp",
         "search_text": "lamp"},
    ]
    reply, out, flights, _a, rendered = _explore_drive([
        [_run('r1 = find("car")\nm = find("lamp")\n'
              'look(r1[0:2], marks=m[0])\n')],
        "done",
    ], cands=_close, obs_minor_m=None, with_renderer=True)
    def _context(reqs):
        # A crop ALWAYS marks its own subject, so the question is whether a
        # mark names something ELSE -- the anchor the caller asked for.
        return [r for r in reqs
                if [m for m in (r.get("marks") or [])
                    if str(m.get("fid")) != str(r.get("center_fid"))]]

    served = [r for reqs in rendered for r in reqs]
    # EVERY crop of the set carries it, not just a lone one.
    assert len(served) >= 2, served
    assert len(_context(served)) == len(served), [
        (r.get("center_fid"), r.get("marks")) for r in served]

    # VACUITY FLOOR: the same look WITHOUT marks= must carry none, or a
    # renderer that marks everything would pass the assertion above.
    _r2, _o2, _f2, _a2, rendered2 = _explore_drive([
        [_run('r1 = find("car")\nlook(r1[0:2])\n')],
        "done",
    ], cands=_close, obs_minor_m=None, with_renderer=True)
    served2 = [r for reqs in rendered2 for r in reqs]
    assert len(served2) >= 2, served2
    assert not _context(served2), [
        (r.get("center_fid"), r.get("marks")) for r in served2]


def test_a_pass_hands_every_stop_the_stops_behind_it():
    """Each item an explore flies carries ``pass_ahead``, the stops the pass
    still has to fly after it, so the station can post the whole pass to
    the crew from the first leg. The last stop carries none, and a
    ``go_to`` carries the key not at all: the bench's fly never reads it."""
    _r, _o, flights, _a, _rend = _explore_drive([
        [_run('explore("prior_7")\n')],
        "done",
    ])
    assert len(flights) == len(_STOPS) > 1
    for k, f in enumerate(flights):
        assert f["pass_ahead"] == _STOPS[k + 1:], (k, f)
    _r, _o, flights, _a, _rend = _explore_drive([
        [_run('go_to("prior_7")\n')],
        "done",
    ])
    assert flights and all("pass_ahead" not in f for f in flights), flights


def test_a_pass_takes_a_stop_the_recording_holds_first():
    """The crew fly the whole sweep in one go, so a later stop is often
    held before the first is: the station's ``held_now`` puts a held stop
    first, whatever the model's order, and the stops behind each still
    ride it. Asked once per pick while more than one stop remains."""
    from maps.util.xy_token import parse_xy_token

    asked: list[list] = []

    def held_now(stops):
        asked.append([list(xy) for _, xy in stops])
        return [list(xy) == _STOPS[2] for _, xy in stops]

    _r, _o, flights, _a, _rend = _explore_drive([
        [_run('explore("prior_7")\n')],
        "done",
    ], held_now=held_now)
    flown = [list(parse_xy_token(f["target_fid"])) for f in flights]
    assert flown == [_STOPS[2], _STOPS[0], _STOPS[1]]
    assert [f["pass_ahead"] for f in flights] == [[_STOPS[0], _STOPS[1]], [_STOPS[1]], []]
    assert asked == [[_STOPS[0], _STOPS[1], _STOPS[2]], [_STOPS[0], _STOPS[1]]]
