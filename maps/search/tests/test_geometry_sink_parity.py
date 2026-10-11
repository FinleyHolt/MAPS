from __future__ import annotations

import json

from maps.search.analyze_scene.geom_program import (
    compact_results, run_program)

# Two buildings in contact, one detached 30 m away, one road with a
# stored centerline (the chart / auto-attach subject).
_B1 = {"fid": "prior_1", "name": "One", "xy": [9.0, 9.0],
       "mask": [[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0], [0.0, 0.0]]],
       "mask_centroid": [10.0, 10.0], "tags": {"building": "yes"}}
_B2 = {"fid": "prior_2", "name": "Two", "xy": [31.0, 11.0],
       "mask": [[[20.0, 0.0], [40.0, 0.0], [40.0, 20.0], [20.0, 20.0], [20.0, 0.0]]],
       "mask_centroid": [30.0, 10.0], "tags": {"building": "yes"}}
_B3 = {"fid": "prior_3", "name": "Three", "xy": [55.0, 10.0],
       "mask": [[[50.0, 0.0], [60.0, 0.0], [60.0, 20.0], [50.0, 20.0], [50.0, 0.0]]],
       "mask_centroid": [55.0, 10.0], "tags": {"building": "yes"}}
# A way: its exact chain IS its mask (open -- it does not return to its first
# vertex), and it also carries the derived centerline the DECLARED
# frame:{along: ...} chart reads.
_ROAD = {"fid": "road_1", "name": "Main", "xy": [30.0, 30.0],
         "mask": [[[0.0, 30.0], [60.0, 30.0]]],
         "centerline": [[0.0, 30.0], [60.0, 30.0]],
         "tags": {"highway": "residential"}}
_CAR = {"fid": "detected_0", "xy": [45.0, 10.0], "class": "car"}

_PROG = {"ops": [
    {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
    {"ref": "r1", "op": "compare",
     "args": {"items": "r0", "refs": ["prior_1", "road_1"]}},
    {"ref": "r2", "op": "compare",
     "args": {"items": "r0",
              "frame": {"along": "road_1", "toward": "prior_3"}}},
    {"ref": "r3", "op": "compare",
     "args": {"items": "r0",
              "frame": {"toward": "prior_1", "from": [55.0, 10.0]}}},
    {"ref": "r4", "op": "score_by",
     "args": {"items": "r0",
              "emphasis": [{"key": "along road_1", "w": -1}]}},
]}


def _anchors():
    return [dict(_B1, tags=dict(_B1["tags"])),
            dict(_B2, tags=dict(_B2["tags"])),
            dict(_B3, tags=dict(_B3["tags"])),
            dict(_ROAD, tags=dict(_ROAD["tags"]),
                 centerline=[list(p) for p in _ROAD["centerline"]])]


def _wire(out):
    return json.dumps(compact_results(out["results"]), default=str)


def test_model_payload_byte_identical_with_and_without_sink():
    bare = run_program([dict(_CAR)], _anchors(), _PROG)
    sink: dict = {}
    sunk = run_program([dict(_CAR)], _anchors(), _PROG, geometry_sink=sink)
    assert _wire(bare) == _wire(sunk)
    for ref in bare["results"]:
        assert list(bare["results"][ref]) == list(sunk["results"][ref])
        assert (getattr(bare["results"][ref], "notes", None)
                == getattr(sunk["results"][ref], "notes", None))
    assert set(bare) == set(sunk) == {"results"}
    assert sink["ops"]  # the sink actually filled


def test_sink_holds_the_resolved_geometry():
    sink: dict = {}
    out = run_program([dict(_CAR)], _anchors(), _PROG, geometry_sink=sink)
    by_ref: dict[str, list] = {}
    for e in sink["ops"]:
        by_ref.setdefault(e["ref"], []).append(e)

    # r1: EVERY row ref resolves to gap witnesses + centroid, road_1
    # included.
    (e1,) = by_ref["r1"]
    centers = {c["label"]: c["center_xy"] for c in e1["ref_centers"]}
    assert centers["prior_1"] == [10.0, 10.0]
    rows = {r.get("fid"): r for r in out["results"]["r1"]}
    # Keyed by (ref label, row fid): every ref produces a gap witness for
    # every row, road_1 included, so a fid-only key collides across refs.
    gaps = {(g["label"], g["fid"]): g for g in e1["gaps"]}
    assert {fid for _lab, fid in gaps} == {"prior_1", "prior_2",
                                           "prior_3", "road_1"}
    g3 = gaps[("prior_1", "prior_3")]
    assert g3["gap_m"] == rows["prior_3"]["distance to prior_1"] == 30.0
    assert g3["a_xy"][0] == 20.0 and g3["b_xy"][0] == 50.0
    assert (gaps[("prior_1", "road_1")]["gap_m"]
            == rows["road_1"]["distance to prior_1"] == 10.0)
    # road_1 as a REF gaps too.
    assert ("road_1", "prior_3") in gaps

    # r2: the chart -- oriented centerline, end0 flag, declared travel.
    (e2,) = by_ref["r2"]
    chart = e2["chart"]
    assert chart["label"] == "road_1"
    assert chart["end0_xy"] == chart["pts"][0]
    assert sorted(map(tuple, chart["pts"])) == sorted(
        map(tuple, _ROAD["centerline"]))
    assert chart["travel"] == {"declared": True, "toward_label": "prior_3"}

    # r3: the viewer frame -- standpoint, look target, basis.
    (e3,) = by_ref["r3"]
    fr = e3["frame"]
    assert fr["type"] == "viewer"
    assert fr["standpoint_xy"] == [55.0, 10.0]
    assert fr["look_xy"] == [9.0, 9.0]  # prior_1's xy (as_point resolution)
    assert fr["origin"] is not None
    assert fr["right_unit"] is not None and fr["up_unit"] is not None

    # r4's emphasis 'along road_1' auto-attaches a chart re-entry; its
    # travel is undeclared (the stored end0->end1 rides).
    autos = by_ref["_autoattach"]
    assert any(e["chart"] and e["chart"]["label"] == "road_1"
               and e["chart"]["travel"]["declared"] is False for e in autos)
