"""The measuring call served on the Python surface: distance().

A served row carries ONE point, and every distance computed off that point
is wrong for an extended feature. These pin the four properties the surface
now depends on:

* gap is FOOTPRINT gap, not centroid distance,
* a SET is one place and means its NEAREST member,
* the number the model reads is the number the executor's own
  ``distance to <ref>`` column carries, and
* a union's geometry is built ONCE, so ``[distance(h, road) for h in houses]``
  is linear in the houses rather than quadratic in the ways.
"""

import math

import pytest

from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.geom_program import run_program
from maps.search.parse_compile.code_surface import CodeError
from maps.search.parse_compile.python_surface import (
    MissionEnv, MissionPython, Row, Rows, _places)

# The headline case, at its measured size: Poppy Boulevard way 5 is 1,198 m
# long and the car parked against it is 7.0 m away.
ROAD = {"fid": "prior_1", "name": "Poppy Boulevard", "class": "road",
        "mask": [[[0.0, 0.0], [0.0, 1198.0]]], "xy": [0.0, 599.0]}
CAR = {"fid": "detected_1", "class": "car", "xy": [7.0, 0.0]}


def _mp(surface: dict, refs: dict | None = None) -> MissionPython:
    """A MissionPython over a fixed surface. Only the geometry hooks are
    live: nothing here dispatches a plan."""
    refs = refs or {}
    env = MissionEnv(
        run_plan=lambda plan, said: {},
        rows_of_ref=lambda ref: refs.get(ref, []),
        plan_rn=[0], discourse={}, minted=[],
        say_err=lambda out, msg: {**out, "error": msg},
        row_of_fid=surface.get)
    return MissionPython(env)


def _served(row: dict) -> dict:
    """A row as an OP OUTPUT serves it: the mask is stripped, so the fid is
    the only way back to the geometry."""
    return {k: v for k, v in row.items() if k != "mask"}


def test_distance_is_footprint_not_centroid():
    mp = _mp({"prior_1": ROAD, "detected_1": CAR})
    assert mp._call_distance("detected_1", "prior_1") == pytest.approx(7.0)
    # ...and the number a centroid read would have produced is nowhere near
    # it, so a regression to point math fails loud rather than passing on a
    # coincidence.
    centroid = math.dist(CAR["xy"], ROAD["xy"])
    assert centroid >= 400.0, centroid


def test_a_set_is_its_nearest_member():
    """The union's chains are ONE geometry, so the min-gap witness picks the
    nearest member. A mean or a first-member read lands 400 m out."""
    near = {"fid": "detected_1", "class": "pool", "xy": [7.0, 0.0]}
    far = {"fid": "detected_2", "class": "pool", "xy": [400.0, 0.0]}
    surface = {"prior_1": ROAD, "detected_1": near, "detected_2": far}
    # the FAR one first, so a first-member read is wrong too
    mp = _mp(surface, refs={"r0": [_served(far), _served(near)]})
    pools = Rows("r0", mp)
    assert len(pools) == 2
    assert mp._call_distance(pools, "prior_1") == pytest.approx(7.0)
    # the alternatives it must not be
    assert mp._call_distance(far["fid"], "prior_1") == pytest.approx(400.0)
    mean_x = (7.0 + 400.0) / 2
    assert abs(mp._call_distance(pools, "prior_1") - mean_x) > 190.0
    # a plain list of Rows/Row unions the same way
    assert mp._call_distance([pools, Row(far, mp=mp)], ROAD["fid"]) \
        == pytest.approx(7.0)


def test_distance_matches_the_executor_column():
    """THE PARITY PIN. The model must be told the number the harness scores
    on: distance() and the compare op's ``distance to <ref>`` are one quantity."""
    out = run_program(
        [dict(CAR)], [dict(ROAD)],
        {"ops": [{"ref": "r0", "op": "compare",
                  "args": {"items": [dict(CAR)], "refs": [ROAD["fid"]]}}]})
    row = out["results"]["r0"][0]
    column = row[f"distance to {ROAD['fid']}"]
    mp = _mp({"prior_1": ROAD, "detected_1": CAR})
    served = mp._call_distance("detected_1", "prior_1")
    assert abs(served - column) <= 0.1, (served, column)
    # VACUITY FLOOR: the executor's OWN centroid column is a different
    # number, so the parity above is not two ways of reading one default.
    assert row[f"center distance to {ROAD['fid']}"] - column > 400.0


def test_union_geometry_is_built_once(monkeypatch):
    """A 64-way union costs ONE mask_geom, however many things are measured
    against it. Without the cache each gap rebuilds all 64 ways."""
    ways = {f"prior_{i}": {"fid": f"prior_{i}", "name": "Neuralia Road",
                           "mask": [[[0.0, i * 20.0], [0.0, i * 20.0 + 20.0]]],
                           "xy": [0.0, i * 20.0 + 10.0]}
            for i in range(64)}
    houses = {f"detected_{i}": {"fid": f"detected_{i}", "class": "house",
                                "xy": [50.0 + i, i * 20.0 + 10.0]}
              for i in range(64)}
    surface = {**ways, **houses}
    mp = _mp(surface)

    calls = {"n": 0}
    real = G.mask_geom

    def counted(chains):
        calls["n"] += 1
        return real(chains)

    monkeypatch.setattr(G, "mask_geom", counted)

    road = list(ways)
    gaps = [mp._call_distance(f"detected_{i % 64}", road) for i in range(100)]
    assert len(gaps) == 100 and all(math.isfinite(g) for g in gaps)
    assert calls["n"] <= 65, calls["n"]
    # VACUITY FLOOR: it really did build the union and the houses, so a
    # cache that returned a stale empty geometry would not pass here.
    assert calls["n"] >= 2, calls["n"]
    assert gaps[0] == pytest.approx(50.0)


def test_a_bare_name_is_refused_rather_than_grounded():
    mp = _mp({"prior_1": ROAD})
    with pytest.raises(CodeError) as e:
        _places("Poppy Boulevard", "gap")
    assert "find(" in str(e.value) and "Poppy Boulevard" in str(e.value)
    # a fid-shaped string is a place, not a name
    assert _places("prior_1", "gap") == (("prior_1", None),)
    assert _places([10.0, 20.0], "gap") == ((None, (10.0, 20.0)),)
    with pytest.raises(CodeError) as e:
        mp._call_distance(3, "prior_1")
    assert "int" in str(e.value)


def test_the_measuring_ceiling_names_the_pairwise_scan(monkeypatch):
    monkeypatch.setattr(
        "maps.search.parse_compile.python_surface.MEASURE_CALL_CEILING", 3)
    mp = _mp({"prior_1": ROAD, "detected_1": CAR})
    for _ in range(3):
        mp._call_distance("detected_1", "prior_1")
    with pytest.raises(CodeError) as e:
        mp._call_distance("detected_1", "prior_1")
    assert "per PAIR" in str(e.value)
    # a fresh block gets a fresh budget
    mp.run_block("")
    assert mp._call_distance("detected_1", "prior_1") == pytest.approx(7.0)
