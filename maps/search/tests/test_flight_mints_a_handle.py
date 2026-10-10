"""A flight returns an ADDRESSABLE set, not just a count.

Standing over what it just found, the model could name none of it: the
flight report carried ``new_detections: 127`` and no handle, so reaching any
of those rows meant re-issuing ``find <noun>.`` over the whole accumulated
world and hoping the new ones ranked into the head. Measured on pilot-10:
126 of 300 episodes reason explicitly about a handle the flight never named.

A view mints the set it shows. These tests drive the real mission loop over
a graph the fly callback actually grows, and assert the minted handle is an
ORDINARY ``rN`` — the next sentence reaches it by name, ``of those,`` reaches
it as the set in scope, and ``look at them.`` frames it.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

from maps.search.analyze_scene.geom_program import build_surface_from_graph
from maps.search.parse_compile.reasoner_loop import (
    MissionSpec,
    advance_reasoner_one_round,
)
from maps.search.tests.test_mission_loop import FakeReasoner


def _run(code: str):
    return ("run", {"code": code})


class _Node:
    """The attributes ``build_surface`` reads off a graph node."""

    def __init__(self, xy, class_name, provenance="detected"):
        self.world_xy = [float(xy[0]), float(xy[1])]
        self.centroid = (int(xy[1]), int(xy[0]))
        self.centerline = None
        self.class_name = class_name
        self.name = None
        self.provenance = provenance


class _Road(_Node):
    """A prior with a baked centerline, so ``crossings`` can construct a
    place from two of them (cell -> world is the cell CENTRE at cell_m 1)."""

    def __init__(self, cells, name):
        (r0, c0) = cells[len(cells) // 2]
        super().__init__((c0 + 0.5, r0 + 0.5), "road", provenance="prebaked")
        self.centerline = list(cells)
        self.name = name
        self.axis_length_m = float(len(cells))


class _Graph:
    def __init__(self, nodes):
        self.nodes = dict(nodes)


_PRIORS = {
    "prior_7": _Node((30.0, 30.0), "depot", provenance="prebaked"),
    "prior_1": _Node((80.0, 80.0), "gate", provenance="prebaked"),
}

# Two roads that MEET at one place — the only sentence shape that both
# builds a set and flies in the same plan.
_ROADS = {
    "road_1": _Road([(100, c) for c in range(80, 121)], "Long Road"),
    "road_2": _Road([(r, 100) for r in range(80, 121)], "Cross Street"),
}


def _drive(script, *, per_flight=3, flights_add=True, evict_on_flight=None,
           extra_priors=None):
    """One mission over a graph the flight GROWS. Returns
    (tool results, the graph, the fids each flight added).

    ``evict_on_flight`` = {flight_index: [fid, ...]} deletes those nodes
    from the graph on that flight, so the surface refresh drops them while
    an earlier handle still holds them."""
    graph = _Graph({**_PRIORS, **(extra_priors or {})})
    added: list[list[str]] = []

    def _fly(item):
        for _dead in (evict_on_flight or {}).get(len(added)) or []:
            graph.nodes.pop(_dead, None)
        new: list[str] = []
        if flights_add:
            base = len(graph.nodes)
            for i in range(per_flight):
                fid = f"detected_{base + i}"
                # one black, the rest white — so a colour follow-up over the
                # minted set has something to separate. Each flight lays its
                # rows on its OWN line: build_surface drops a co-located
                # duplicate, so a second flight over the same points would
                # add no rows at all.
                graph.nodes[fid] = _Node(
                    (100.0 + 6.0 * i, 100.0 + 20.0 * len(added)),
                    "black car" if i == 0 else "white car")
                new.append(fid)
        added.append(new)
        return {"stop_id": f"stop_{len(added) - 1:03d}",
                "target": item["target_fid"], "pose_xy": [100.0, 100.0],
                "alt_m": 144.8, "new_detections": len(new), "cue": "the cue"}

    cands, anchors = build_surface_from_graph(
        graph.nodes.items(), (0.0, 0.0), 1.0)
    fr = FakeReasoner(script)
    advance_reasoner_one_round(
        reasoner=fr,
        user_message="Find the goal object.",
        ctx=SimpleNamespace(graph=graph, cell_size_m=1.0,
                            tile_origin_xy=(0.0, 0.0), grid_shape=(10, 10),
                            robot_pose=None),
        rid_tag="m0",
        transcript=[],
        compute_namespace={"cands": cands, "anchors": anchors,
                           "cell_m": 1.0},
        diag={},
        look_renderer=lambda reqs: [
            {"jpeg": b"x", "center_fid": r["center_fid"], "legend": [
                {"digit": 1, "fid": r["center_fid"], "label": "car",
                 "role": "center"}] + [
                {"digit": 3 + i, "fid": m["fid"], "label": "car",
                 "role": "context"}
                for i, m in enumerate(r["marks"])
                if m["fid"] != r["center_fid"]],
             "context_m": 30.0, "world_window": [0, 0, 200, 200],
             "marks_dropped": 0, "marks_out_of_view": [],
             "marks_unbadged_in_frame": []}
            for r in reqs],
        mission_spec=MissionSpec(fly=_fly, declare_concepts=lambda i: {
            "declared": []}),
    )
    results = [json.loads(m["content"]) for m in fr.messages
               if m.get("role") == "tool"]
    return results, graph, added


def _flown(result: dict) -> dict:
    (rep,) = result["flown"]["flown"]
    return rep


def _minted_handle(script, *, nth=0, **kw) -> str:
    """The handle the ``nth`` flight of ``script`` minted.

    The number rides the mission's shared ref counter, so a block that ran
    an analyze before flying gets a higher one. Tests learn it the way the
    model does -- off the flight's own report -- instead of assuming r0.
    """
    probe, _g, _a = _drive(list(script) + ["done"], **kw)
    flown = [r for r in probe if isinstance(r, dict) and "flown" in r]
    return _flown(flown[nth])["new_rows"].split(" ")[0]


def _fids(result: dict) -> list[str]:
    """The terminal op's served fids, in its own order."""
    envs = list(result["result"]["results"].values())
    return [str(f) for f in (envs[-1].get("fids") or [])]


def _shown_fids(result: dict) -> list[str]:
    """The fids a block SHOWED, in order.

    A predicate filter is ordinary Python over rows already served, so its
    answer comes back in ``printed`` rather than as an op envelope. Both are
    the block's served output; this reads whichever one carries rows.
    """
    if "result" in result:
        return _fids(result)
    return re.findall(r"fid='([^']+)'", str(result.get("printed") or ""))


def test_the_flight_names_the_rows_it_added(_n=3):
    results, _graph, added = _drive(
        [[_run('go_to("prior_7")')], "done"], per_flight=_n)
    rep = _flown(results[0])
    assert added[0] and len(added[0]) == _n, added
    # the count was always there; the ADDRESS is what was missing
    assert rep["new_detections"] == _n, rep
    handle = rep["new_rows"].split(" ")[0]
    assert handle.startswith("r") and handle[1:].isdigit(), rep
    assert f"the {_n} rows this flight added" in rep["new_rows"], rep
    # a fact and a capability, never an imperative sequence -- and in the
    # register of the surface being served, which is the code arm's
    assert (f"the handle is bound, so {handle}, {handle}[0] and "
            f"{handle}.order(...) all reach them") in rep["new_rows"], rep


def test_the_minted_handle_is_bound_for_the_next_block():
    # The flight's own rows are reachable by NAME in the very next block,
    # with no intervening find — the thing the model could not do before.
    handle = _minted_handle([[_run('go_to("prior_7")')]])
    results, _graph, added = _drive(
        [[_run('go_to("prior_7")')],
         [_run(f'b = {handle}.order(color="black")\nb')],
         "done"])
    rep = _flown(results[0])
    assert rep["new_rows"].split(" ")[0] == handle, rep

    out = results[1]
    assert "error" not in out, out
    fids = _shown_fids(out)
    # every row came from THIS flight
    assert fids and fids == added[0], (fids, added[0])


def test_the_minted_handle_gets_a_crop_per_row(_n=4):
    results, _graph, added = _drive(
        [[_run('go_to("prior_7")')], [_run("look(r0)")], "done"],
        per_flight=_n)
    crop = results[1]["crop"]
    assert "error" not in crop, crop
    # one crop PER ROW now, not one frame over the set
    assert crop["delivered"] == crop["looked"] >= 1, crop
    assert "badged" not in crop, crop      # per-row crops, no frame badges
    assert crop["delivered"] == _n, crop
    handle = _flown(results[0])["new_rows"].split(" ")[0]
    # the per-row form has no frame to count badges in; the served
    # fact is how many crops came back.
    assert crop["delivered"] == _n, crop


def test_a_flight_that_adds_nothing_mints_nothing():
    # An empty handle that reads as live is this repo's named hazard, and
    # rule T's own rule is non-empty-only: a zero-detection flight leaves
    # the scope exactly where it was.
    results, _graph, added = _drive(
        [[_run('go_to("prior_7")')], [_run("look(r0)")], "done"],
        flights_add=False)
    rep = _flown(results[0])
    assert added == [[]], added
    assert "new_rows" not in rep, rep
    err = str(results[1].get("error"))
    assert "NameError" in err and "'r0'" in err, results[1]


def test_a_row_gone_from_the_surface_is_not_reported_as_out_of_frame():
    # A requested mark goes undrawn for four different reasons and only ONE
    # of them is distance. Here a row the handle still holds is gone from
    # the live surface, so _build_crop_requests SKIPS it before the renderer
    # ever sees it: it is not out of frame, the vision legend never names
    # it, and it may be metres from the subject. A requested-minus-drawn
    # residual would report it as "outside the frame" -- the same false
    # precision the drawn count was added to remove.
    results, _graph, added = _drive(
        [[_run('go_to("prior_7")')], [_run('go_to("prior_1")')],
         [_run("look(r0)")], "done"],
        evict_on_flight={1: ["detected_3"]})   # rank 1 of r0, evicted
    crop = results[2]["crop"]
    # VACUITY FLOOR: the eviction must actually have reached the crop path.
    assert crop["skipped"] == ["detected_3"], crop
    # "badged" belonged to the single-frame form; every row now has its
    # own crop, so what is served is the count DELIVERED.
    assert "badged" not in crop, crop
    # one row is gone from the surface, so it is SKIPPED, not delivered
    assert crop["delivered"] == len(added[0]) - len(crop["skipped"]), crop
    assert "sit outside the frame" not in crop["note"], crop["note"]


_XING_BLOCK = ('x = intersection(find("Long Road"), find("Cross Street"))\n'
               'go_to(x[0])')


def test_a_block_that_both_builds_a_set_and_flies_ends_on_the_flight():
    # THE case the post-update_discourse re-apply exists for: a block that
    # runs an analyze step (the crossings op, which sets last_ref) and THEN
    # flies. update_discourse would leave the scope on the crossings rows;
    # the flight is the most recent thing that happened, so it takes the
    # scope back. Without the re-apply the next block works over crossings,
    # which carry no colour, and refuses.
    handle = _minted_handle([[_run(_XING_BLOCK)]], extra_priors=_ROADS)
    results, _graph, added = _drive(
        [[_run(_XING_BLOCK)],
         [_run(f'b = {handle}.order(color="black")\nb')],
         "done"],
        extra_priors=_ROADS)
    # VACUITY FLOOR: the sentence must really have built a set AND flown,
    # or the re-apply is not on this path at all.
    assert "result" in results[0] and "flown" in results[0], results[0]
    rep = _flown(results[0])
    assert rep["new_rows"].split(" ")[0] == handle, rep
    # the crossings set keeps the name the BLOCK bound, and the flight's
    # minted handle is a different one -- the flight did not overwrite it
    bound = results[0]["handles"]
    assert "x" in bound, results[0]
    assert handle in bound and handle != "x", (bound, handle)

    out = results[1]
    assert "error" not in out, out
    assert _shown_fids(out) == added[0], (out, added[0])
    # the crossings set is not lost -- it keeps its own handle
    results2, _g2, _a2 = _drive(
        [[_run(_XING_BLOCK)], [_run("look(x)")], "done"],
        extra_priors=_ROADS)
    assert "error" not in results2[1], results2[1]
    assert results2[1]["crop"]["delivered"] == 1, results2[1]


def test_the_latest_flight_owns_the_scope():
    # The flight is the most recent thing that happened, so it takes
    # "those" from whatever held it — an earlier find over the whole world,
    # or an earlier flight. Both older sets keep their own handles.
    _prefix = [[_run('go_to("prior_7")')], [_run('find("car")')],
               [_run('go_to("prior_1")')]]
    second_handle = _minted_handle(_prefix, nth=1)
    results, _graph, added = _drive(
        _prefix + [[_run(f'b = {second_handle}.order(color="black")\nb')],
                   "done"])
    first, second = added[0], added[1]
    # vacuity floor: two DISTINCT flight sets, and the find in between held
    # "those" over the first of them — so the second flight really moved it.
    assert first and second and not (set(first) & set(second)), added
    assert _fids(results[1]) == first, (results[1], first)

    assert _shown_fids(results[3]) == second, (results[3], second)
    # A FIND HANDLE IS A QUERY, so naming it after a later flight answers
    # about the world THAT flight folded -- the find's rows re-derive, and
    # the cars the second flight added are in them. The snapshot semantics
    # this replaced is what let calcity::200#198 commit out of a red-car
    # list bound before the ground the target sat on had been flown.
    find_handle = list(results[1]["result"]["results"])[-1]
    results2, _g2, added2 = _drive(
        [[_run('go_to("prior_7")')], [_run('find("car")')],
         [_run('go_to("prior_1")')],
         [_run(f'b = {find_handle}.order(color="black")\nb')], "done"])
    got = set(_fids(results2[3]))
    assert got >= set(added2[0]), (got, added2[0])
    # ...and it reaches the SECOND flight's rows, which the snapshot could
    # not: that is the whole point of re-deriving.
    assert got & set(added2[1]), (got, added2[1])
    # (The flight handles still own their own scope -- asserted above on
    # results[1] / results[3]: an event is not a query.)
