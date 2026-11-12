"""A flight returns an ADDRESSABLE set, not just a count."""

from __future__ import annotations

import re

from maps.mission.tests.fakes import ROADS, grow_drive, run


def _flown(result: dict) -> dict:
    (rep,) = result["flown"]["flown"]
    return rep


def _minted_handle(script, *, nth=0, **kw) -> str:
    """The handle the ``nth`` flight of ``script`` minted."""
    probe, _g, _a = grow_drive(list(script) + ["done"], **kw)
    flown = [r for r in probe if isinstance(r, dict) and "flown" in r]
    return _flown(flown[nth])["new_rows"].split(" ")[0]


def _fids(result: dict) -> list[str]:
    """The terminal op's served fids, in its own order."""
    envs = list(result["result"]["results"].values())
    return [str(f) for f in (envs[-1].get("fids") or [])]


def _shown_fids(result: dict) -> list[str]:
    """The fids a block SHOWED, in order."""
    if "result" in result:
        return _fids(result)
    return re.findall(r"fid='([^']+)'", str(result.get("printed") or ""))


def test_the_flight_names_the_rows_it_added(_n=3):
    results, _graph, added = grow_drive([[run('go_to("prior_7")')], "done"], per_flight=_n)
    rep = _flown(results[0])
    assert added[0] and len(added[0]) == _n, added
    # the report carries the count AND the handle that addresses the rows
    assert rep["new_detections"] == _n, rep
    handle = rep["new_rows"].split(" ")[0]
    assert handle.startswith("r") and handle[1:].isdigit(), rep
    assert f"the {_n} rows this flight added" in rep["new_rows"], rep
    # a fact and a capability, never an imperative sequence -- and in the
    # register of the surface being served, which is the code arm's
    assert (
        f"the handle is bound, so {handle}, {handle}[0] and {handle}.order(...) all reach them"
    ) in rep["new_rows"], rep


def test_the_minted_handle_is_bound_for_the_next_block():
    # The flight's own rows are reachable by NAME in the very next block,
    # with no intervening find — the thing the model could not do before.
    handle = _minted_handle([[run('go_to("prior_7")')]])
    results, _graph, added = grow_drive(
        [[run('go_to("prior_7")')], [run(f'b = {handle}.order(color="black")\nb')], "done"]
    )
    rep = _flown(results[0])
    assert rep["new_rows"].split(" ")[0] == handle, rep

    out = results[1]
    assert "error" not in out, out
    fids = _shown_fids(out)
    # every row came from THIS flight
    assert fids and fids == added[0], (fids, added[0])


def test_the_minted_handle_gets_a_crop_per_row(_n=4):
    results, _graph, added = grow_drive(
        [[run('go_to("prior_7")')], [run("look(r0)")], "done"], per_flight=_n
    )
    crop = results[1]["crop"]
    assert "error" not in crop, crop
    assert crop["delivered"] == crop["looked"] >= 1, crop
    assert "badged" not in crop, crop  # per-row crops, no frame badges
    assert crop["delivered"] == _n, crop


def test_a_flight_that_adds_nothing_mints_nothing():
    # An empty handle that reads as live is this repo's named hazard, and
    # the scope moves only to a non-empty set: a zero-detection flight leaves
    # the scope exactly where it was.
    results, _graph, added = grow_drive(
        [[run('go_to("prior_7")')], [run("look(r0)")], "done"], flights_add=False
    )
    rep = _flown(results[0])
    assert added == [[]], added
    assert "new_rows" not in rep, rep
    err = str(results[1].get("error"))
    assert "NameError" in err and "'r0'" in err, results[1]


def test_a_row_gone_from_the_surface_is_not_reported_as_out_of_frame():
    # A requested mark goes undrawn for four different reasons and only ONE
    # of them is distance. Here a row the handle still holds is gone from
    # the live surface, so views.crop_requests SKIPS it before the renderer
    # ever sees it: it is not out of frame, the vision legend never names
    # it, and it may be metres from the subject. A requested-minus-drawn
    # residual would report it as "outside the frame".
    results, _graph, added = grow_drive(
        [[run('go_to("prior_7")')], [run('go_to("prior_1")')], [run("look(r0)")], "done"],
        evict_on_flight={1: ["detected_3"]},
    )  # rank 1 of r0, evicted
    crop = results[2]["crop"]
    # VACUITY FLOOR: the eviction must actually have reached the crop path.
    assert crop["skipped"] == ["detected_3"], crop
    assert "badged" not in crop, crop
    # one row is gone from the surface, so it is SKIPPED, not delivered
    assert crop["delivered"] == len(added[0]) - len(crop["skipped"]), crop
    assert "sit outside the frame" not in crop["note"], crop["note"]


_XING_BLOCK = 'x = intersection(find("Long Road"), find("Cross Street"))\ngo_to(x[0])'


def test_a_block_that_both_builds_a_set_and_flies_ends_on_the_flight():
    # THE case the post-update_discourse re-apply exists for: a block that
    # runs an analyze step (the crossings op, which sets last_ref) and THEN
    # flies. update_discourse would leave the scope on the crossings rows;
    # the flight is the most recent thing that happened, so it takes the
    # scope back. Without the re-apply the next block works over crossings,
    # which carry no colour, and refuses.
    handle = _minted_handle([[run(_XING_BLOCK)]], extra_priors=ROADS)
    results, _graph, added = grow_drive(
        [[run(_XING_BLOCK)], [run(f'b = {handle}.order(color="black")\nb')], "done"],
        extra_priors=ROADS,
    )
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
    results2, _g2, _a2 = grow_drive(
        [[run(_XING_BLOCK)], [run("look(x)")], "done"], extra_priors=ROADS
    )
    assert "error" not in results2[1], results2[1]
    assert results2[1]["crop"]["delivered"] == 1, results2[1]


def test_the_latest_flight_owns_the_scope():
    # The flight is the most recent thing that happened, so it takes
    # "those" from whatever held it — an earlier find over the whole world,
    # or an earlier flight. Both older sets keep their own handles.
    _prefix = [[run('go_to("prior_7")')], [run('find("car")')], [run('go_to("prior_1")')]]
    second_handle = _minted_handle(_prefix, nth=1)
    results, _graph, added = grow_drive(
        _prefix + [[run(f'b = {second_handle}.order(color="black")\nb')], "done"]
    )
    first, second = added[0], added[1]
    # vacuity floor: two DISTINCT flight sets, and the find in between held
    # "those" over the first of them — so the second flight really moved it.
    assert first and second and not (set(first) & set(second)), added
    assert _fids(results[1]) == first, (results[1], first)

    assert _shown_fids(results[3]) == second, (results[3], second)
    # A FIND HANDLE IS A QUERY, so naming it after a later flight answers about the
    # world THAT flight folded -- the find's rows re-derive, and the cars the second
    # flight added are in them.
    find_handle = list(results[1]["result"]["results"])[-1]
    results2, _g2, added2 = grow_drive(
        [
            [run('go_to("prior_7")')],
            [run('find("car")')],
            [run('go_to("prior_1")')],
            [run(f'b = {find_handle}.order(color="black")\nb')],
            "done",
        ]
    )
    got = set(_fids(results2[3]))
    assert got >= set(added2[0]), (got, added2[0])
    # ...and it reaches the SECOND flight's rows, which the snapshot could
    # not: that is the whole point of re-deriving.
    assert got & set(added2[1]), (got, added2[1])
    # (The flight handles still own their own scope -- asserted above on
    # results[1] / results[3]: an event is not a query.)
