"""Shape-natural relation columns: ``compare{refs}`` attaches each ref's
OWN geometry. A POINT or POLYGON ref, the four relation columns — d_<ref>
(footprint gap, 0.0 on touch/overlap), brg_gap_<ref> (compass bearing of the
min-gap vector, ref witness -> item witness; ABSENT exactly when d_<ref>=0 —
on contact no local direction exists), dc_<ref> (centroid-to-centroid
distance) and brg_<ref> (centroid->centroid compass bearing, defined on
contact too; absent only when the centroids coincide, dc_<ref>=0).
A LINE ref (a stored centerline, mask or not) charts along_<ref> (arc
meters from its end0_xy end, UN-CLAMPED past the ends) + off_<ref> (signed
lateral meters, + = right of travel) INSTEAD — a 946 m road's d_ tie-block
at 0.0 is contact, not location. Union refs stay mask aggregates (gap
semantics).
"""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import ProgramError, run_program


def _compare(cands, anchors, refs):
    out = run_program([dict(c) for c in cands], [dict(a) for a in anchors],
                      {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare", "args": {"items": "$r0", "refs": refs}},
    ]})
    return {r["fid"]: r for r in out["results"]["r1"]}


# A long LINEAR road: centerline along x from (0,0) to (100,0), centroid at
# (50,0); no mask (the polyline-only data kind).
_ROAD = {"fid": "road_1", "name": "Main Street", "xy": [50.0, 0.0],
         "class": "road", "centerline": [[0.0, 0.0], [100.0, 0.0]],
         "axis_length_m": 100.0, "axis_residual_m": 2.0}
# A square lot footprint (exact one-loop mask) centered at (10, 10).
_LOT = {"fid": "lot_1", "name": "Lot", "xy": [10.0, 10.0], "class": "parking",
        "mask": [[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0], [0.0, 0.0]]]}


def test_brg_gap_absent_on_contact_brg_still_present():
    # d1 sits INSIDE the lot (d=0: contact -> no local direction, brg_gap_
    # honestly ABSENT); brg_ (centroid->centroid) still reads -- "the south
    # part of the lot". d2 is outside and carries both.
    cands = [
        {"fid": "d1", "xy": [10.0, 5.0], "class": "car"},    # inside the lot
        {"fid": "d2", "xy": [10.0, 30.0], "class": "car"},   # 10m north of it
    ]
    rows = _compare(cands, [_LOT], ["lot_1"])
    inside, outside = rows["d1"], rows["d2"]
    assert inside["distance to lot_1"] == 0.0, inside
    assert "facing lot_1" not in inside, list(inside)
    assert "bearing from lot_1" in inside and "center distance to lot_1" in inside, list(inside)
    # centroid (10,10) -> (10,5): (0,-5) = due south -> ~180 deg.
    assert 170.0 < inside["bearing from lot_1"] < 190.0, inside
    assert outside["distance to lot_1"] == 10.0, outside
    assert "facing lot_1" in outside, list(outside)
    # locally due north of the lot's wall ((0,+10) -> ~0 deg).
    # A WORD, and the REVERSE of where the row LIES: the row sits north of
    # the lot, so it FACES south at it. That inversion used to be a formula
    # in the legend and the one episode that read the degrees applied it
    # zero times (calcity::480#478).
    assert outside["facing lot_1"] == "south", outside


def test_point_ref_carries_all_four_columns():
    cands = [{"fid": "d0", "xy": [3.0, 4.0], "class": "car"}]
    rows = _compare(cands, [], [[0.0, 0.0]])
    r = rows["d0"]
    assert r["distance to xy"] == 5.0 and r["center distance to xy"] == 5.0, r
    # a point ref's witness IS its centroid: brg_gap_ == brg_ off contact.
    # 'bearing from' stays a centroid-to-centroid NUMBER, ref->item;
    # 'facing' is the footprint WORD, item->ref. Opposites by construction,
    # which is the whole reason the word is served instead of the degrees.
    from maps.search.analyze_scene import geom_compute as G
    assert r["facing xy"] == G._compass8((r["bearing from xy"] + 180) % 360), r


def test_mask_gap_is_the_only_d_path():
    # two footprints touching -> d 0.0 (contact), brg_gap_ absent; the item's
    # own mask (not its centroid) is the measured side.
    cands = [{"fid": "d0", "xy": [25.0, 10.0], "class": "car",
              "mask": [[[20.0, 8.0], [30.0, 8.0], [30.0, 12.0], [20.0, 12.0], [20.0, 8.0]]]}]
    rows = _compare(cands, [_LOT], ["lot_1"])
    r = rows["d0"]
    assert r["distance to lot_1"] == 0.0, r            # wall touches the lot edge
    assert "facing lot_1" not in r, list(r)
    assert "bearing from lot_1" in r, list(r)


def test_multi_fragment_mask_measures_every_fragment():
    # The EXACT mask keeps every disjoint fragment: a candidate near the
    # SMALL fragment reads its gap from that fragment, never from the big
    # loop across the map (the dropped-fragment failure mode).
    frag_ref = {"fid": "building_9", "name": "One Stop", "xy": [50.0, 0.0],
                "class": "building",
                "mask": [
                    [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0], [0.0, 0.0]],
                    [[100.0, 0.0], [104.0, 0.0], [104.0, 4.0], [100.0, 4.0], [100.0, 0.0]],
                ]}
    cands = [{"fid": "d0", "xy": [106.0, 2.0], "class": "car"}]
    r = _compare(cands, [frag_ref], ["building_9"])["d0"]
    assert r["distance to building_9"] == 2.0, r      # 2 m off the small fragment
    # and a candidate INSIDE the small fragment is contact
    cands = [{"fid": "d1", "xy": [102.0, 2.0], "class": "car"}]
    r = _compare(cands, [frag_ref], ["building_9"])["d1"]
    assert r["distance to building_9"] == 0.0, r


def test_crossing_boundaries_read_zero_gap():
    # Plus-sign overlap: the boundaries CROSS with no vertex of either inside
    # the other -- the interiors intersect, so d_ = 0 and brg_gap_ is absent.
    # The two masks' centroids COINCIDE here, so dc_ = 0 and brg_ is absent
    # too (no direction exists between coincident centroids -- the same
    # degenerate idiom as brg_gap_ at contact).
    wide = {"fid": "lot_2", "name": "Wide", "xy": [0.0, 0.0], "class": "parking",
            "mask": [[[-10.0, -1.0], [10.0, -1.0], [10.0, 1.0], [-10.0, 1.0], [-10.0, -1.0]]]}
    cands = [{"fid": "d0", "xy": [0.0, 0.0], "class": "car",
              "mask": [[[-1.0, -10.0], [1.0, -10.0], [1.0, 10.0], [-1.0, 10.0], [-1.0, -10.0]]]}]
    r = _compare(cands, [wide], ["lot_2"])["d0"]
    assert r["distance to lot_2"] == 0.0, r
    assert "facing lot_2" not in r, list(r)
    assert r["center distance to lot_2"] == 0.0, r
    assert "bearing from lot_2" not in r, list(r)
    # Offset the crossing candidate: still contact (d_=0, brg_gap_ absent)
    # but the centroids now differ, so dc_ > 0 and brg_ IS defined on
    # contact -- the always-on-contact half of the contract.
    cands = [{"fid": "d1", "xy": [4.0, 0.0], "class": "car",
              "mask": [[[3.0, -10.0], [5.0, -10.0], [5.0, 10.0], [3.0, 10.0], [3.0, -10.0]]]}]
    r = _compare(cands, [wide], ["lot_2"])["d1"]
    assert r["distance to lot_2"] == 0.0, r
    assert "facing lot_2" not in r, list(r)
    assert r["center distance to lot_2"] > 0.0 and "bearing from lot_2" in r, r


def test_centerline_row_renders_its_termini():
    # A centerline row carries end0_xy (the along_=0 end of every chart
    # over it) + end1_xy — a terminus/corner cue is derivable from the row
    # itself (nothing served termini before; 4/5 audited spatial failures
    # hand-derived crossings they could not cite).
    from maps.search.analyze_scene.geom_program import build_surface_from_graph

    class _N:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getattr__(self, k):
            return None

    road = _N(name="Main Street", provenance="prebaked",
              world_xy=[50.0, 0.0],
              centerline=[[0.0, 0.0], [60.0, 0.0], [100.0, 4.0]],
              tags={"highway": "residential"})
    lot = _N(name="Lot", provenance="prebaked", world_xy=[10.0, 10.0],
             tags={"amenity": "parking"})
    _, anchors = build_surface_from_graph(
        [("road_1", road), ("lot_1", lot)], (0.0, 0.0), 2.0)
    rows = {a["fid"]: a for a in anchors}
    cl = rows["road_1"]["centerline"]  # node cell-space -> world by _w
    assert rows["road_1"]["end0_xy"] == [round(cl[0][0], 2),
                                         round(cl[0][1], 2)], rows["road_1"]
    assert rows["road_1"]["end1_xy"] == [round(cl[-1][0], 2),
                                         round(cl[-1][1], 2)], rows["road_1"]
    assert "end0_xy" not in rows["lot_1"], rows["lot_1"]  # points have no ends


def test_dc_brg_derive_from_the_mask_centroid_not_from_an_xy():
    # A row ref carrying a mask but NO usable xy still serves dc_/brg_ --
    # the center is the mask's TRUE area centroid (10, 10), derived from the
    # mask itself. The ONE thing that withholds them is geometry that is
    # several disjoint places (geom_compute.chain_components), which this
    # single ring is not.
    bare = {"fid": "lot_3", "name": "Bare", "xy": None, "class": "parking",
            "mask": [[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0], [0.0, 0.0]]]}
    cands = [{"fid": "d0", "xy": [10.0, 40.0], "class": "car"}]
    out = run_program([{"fid": "d0", "xy": [10.0, 40.0], "class": "car"}],
                      [bare], {"ops": [
        {"ref": "r0", "op": "compare",
         "args": {"items": [dict(c) for c in cands], "refs": ["lot_3"]}},
    ]})
    r = {x["fid"]: x for x in out["results"]["r0"]}["d0"]
    assert r["center distance to lot_3"] == 30.0, r          # (10,10) -> (10,40)
    assert "bearing from lot_3" in r, list(r)


def test_cells_realise_exactly_multiloop_and_holes():
    # The raster realisation is the EXACT ingested mask: every fragment kept,
    # holes excluded by even-odd membership, no decimation.
    from maps.search.analyze_scene import geom_compute as G

    # a 3x3 block with the center cell missing (a donut) at cell_m=2
    cells = [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
    loops = G.cells_boundary_loops(cells, (0.0, 0.0), 2.0)
    assert len(loops) == 2, loops            # outer + hole loop, both kept
    g = G.mask_geom(loops)
    # the hole center (3,3) is OUTSIDE the mask: 1 m to the hole wall
    d, _, _ = G.gap_witness(g, G.mask_geom([[[3.0, 3.0]]]))
    assert abs(d - 1.0) < 1e-9, d
    # a point inside the ring reads containment
    d, _, _ = G.gap_witness(g, G.mask_geom([[[1.0, 1.0]]]))
    assert d == 0.0, d
    # two disjoint fragments: BOTH kept; the gap reads from the nearer one
    loops2 = G.cells_boundary_loops([(0, 0), (0, 10)], (0.0, 0.0), 2.0)
    assert len(loops2) == 2, loops2
    d, _, _ = G.gap_witness(G.mask_geom(loops2),
                            G.mask_geom([[[23.0, 1.0]]]))
    assert abs(d - 1.0) < 1e-9, d            # 1 m off the (0,10) fragment
    # the TRUE area centroid subtracts the hole (symmetry: stays the center)
    cx, cy = G.loops_area_centroid(loops)
    assert abs(cx - 3.0) < 1e-9 and abs(cy - 3.0) < 1e-9, (cx, cy)
    # every loop STATES its closure — the trace ends on the vertex it began
    for lp in loops + loops2:
        assert G.is_closed_chain(lp), lp


def test_a_chain_states_its_own_closure():
    """The one rule that tells a ring from a way. Nothing asks what kind of
    feature it belongs to; the same vertices differ only by the closing one."""
    from maps.search.analyze_scene import geom_compute as G

    square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    ring, way = square + [[0.0, 0.0]], list(square)
    assert G.is_closed_chain(ring) and not G.is_closed_chain(way)
    inside = G.mask_geom([[[5.0, 5.0]]])

    # A ring ENCLOSES: its interior is contact, and it has a footprint.
    assert G.gap_witness(G.mask_geom([ring]), inside)[0] == 0.0
    assert bool(G.mask_contains(G.mask_geom([ring]), [[5.0, 5.0]])[0])
    assert G.contour_shape_features([ring])["footprint_area_m2"] == 100.0

    # The SAME vertices as a way enclose nothing: the interior point is 5 m
    # off the nearest edge, containment is false, and there is no footprint.
    # Its missing closing edge is what the difference IS -- a point beyond
    # the open side reads its distance to that side, not zero.
    assert G.gap_witness(G.mask_geom([way]), inside)[0] == 5.0
    assert not G.mask_contains(G.mask_geom([way]), [[5.0, 5.0]])[0]
    assert G.contour_shape_features([way]) is None

    # A single vertex is a point: no segment, no closure, exact position.
    pt = G.mask_geom([[[3.0, 4.0]]])
    assert not pt["closed"] and len(pt["A"]) == 0
    assert G.gap_witness(pt, G.mask_geom([[[0.0, 0.0]]]))[0] == 5.0

    # A side may mix them: distance runs over every edge, containment only
    # over the closed one, so a way on the side never fakes an interior.
    mixed = G.mask_geom([ring, [[40.0, 0.0], [40.0, 10.0]]])
    assert len(mixed["A"]) == 5 and len(mixed["CA"]) == 4
    assert bool(G.mask_contains(mixed, [[5.0, 5.0]])[0])
    assert G.gap_witness(mixed, G.mask_geom([[[42.0, 5.0]]]))[0] == 2.0


def test_only_a_closed_chain_carries_area_or_an_area_centroid():
    """A way encloses nothing, so it has neither area nor an area centroid.

    The shoelace wraps with ``% n``. On a ring that re-walks the duplicated
    closing vertex for a zero term; on an OPEN way it manufactures a phantom
    closing edge. A way that doubles back near-cancels its own phantom lobes,
    so the signed sum lands just above the 1e-12 guard while the moments do
    not cancel at all, and ``sx / (3*sa)`` throws the centre clean off the
    geometry. ``center distance to <ref>`` and ``bearing from <ref>`` are
    SERVED columns computed from that point.
    """
    from maps.search.analyze_scene import geom_compute as G

    square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    ring, way = square + [[0.0, 0.0]], list(square)

    # Same vertices: the ring encloses 100 m2 about (5, 5); the way encloses
    # nothing and reports its vertex mean, which is a position a way has.
    assert G.loops_area([ring]) == 100.0
    assert G.loops_area_centroid([ring]) == [5.0, 5.0]
    assert G.loops_area([way]) == 0.0
    assert G.loops_area_centroid([way]) == [5.0, 5.0]

    # A way that loops one way near its start and the other way at its far
    # end: the phantom areas cancel to a hair, the moments do not.
    doubling = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],
                [0.0, 0.5], [1000.0, 0.0], [1000.0, -10.0],
                [1010.0, -10.0], [1010.0, 0.0]]
    assert G.loops_area([doubling]) == 0.0
    cx, cy = G.loops_area_centroid([doubling])
    xs = [p[0] for p in doubling]
    assert min(xs) <= cx <= max(xs), (cx, cy)   # unguarded: x = -353.3

    # A side may mix them: only the ring's 100 m2 counts, and the centre is
    # the ring's, not a blend of the ring with a phantom.
    mixed = [ring, [[400.0, 0.0], [400.0, 100.0]]]
    assert G.loops_area(mixed) == 100.0
    assert G.loops_area_centroid(mixed) == [5.0, 5.0]


def test_frame_along_is_unclamped_past_the_ends():
    # The polyline chart extends past its ends: a candidate beyond the far
    # end reads its TRUE arc coordinate (> length), one before the start < 0.
    cands = [
        {"fid": "past", "xy": [110.0, -5.0], "class": "car"},
        {"fid": "before", "xy": [-10.0, -5.0], "class": "car"},
        {"fid": "mid", "xy": [50.0, -5.0], "class": "car"},
    ]
    out = run_program([dict(c) for c in cands], [dict(_ROAD)], {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "$r0", "frame": {"along": "road_1"}}},
    ]})
    rows = {r["fid"]: r for r in out["results"]["r1"]}
    assert rows["past"]["along road_1"] == 110.0, rows["past"]
    assert rows["before"]["along road_1"] == -10.0, rows["before"]
    assert rows["mid"]["along road_1"] == 50.0, rows["mid"]
    assert not any(k.startswith("clamped_") for r in rows.values() for k in r)


# --------------------------------------------------------------------------- #
# UNION refs: {ref:[fid,...], as:'label'} -- the named masks as ONE ref        #
# --------------------------------------------------------------------------- #

# Two DISJOINT segments of one named road, 60 m apart (a fragmented
# multi-instance road): each a 20x4 m footprint on y in [-2, 2].
_SEG_A = {"fid": "road_86", "name": "St Peters Street", "xy": [10.0, 0.0],
          "class": "road",
          "mask": [[[0.0, -2.0], [20.0, -2.0], [20.0, 2.0], [0.0, 2.0], [0.0, -2.0]]]}
_SEG_B = {"fid": "road_88", "name": "St Peters Street", "xy": [90.0, 0.0],
          "class": "road",
          "mask": [[[80.0, -2.0], [100.0, -2.0], [100.0, 2.0], [80.0, 2.0], [80.0, -2.0]]]}


def _union_run(cands, extra_ops=None):
    ops = [
        {"ref": "r0", "op": "find",
         "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "$r0",
                  "refs": [{"ref": ["road_86", "road_88"],
                            "as": "st_peters"}]}},
    ] + list(extra_ops or [])
    return run_program([dict(c) for c in cands],
                       [dict(_SEG_A), dict(_SEG_B)], {"ops": ops})


def test_union_d_is_exact_min_gap_over_disjoint_members():
    cands = [
        {"fid": "d0", "xy": [10.0, 10.0], "class": "car"},   # 8 m off SEG_A
        {"fid": "d1", "xy": [90.0, 5.0], "class": "car"},    # 3 m off SEG_B
    ]
    out = _union_run(cands)
    rows = {r["fid"]: r for r in out["results"]["r1"]}
    assert rows["d0"]["distance to st_peters"] == 8.0, rows["d0"]      # nearest member
    assert rows["d1"]["distance to st_peters"] == 3.0, rows["d1"]      # ...each side
    # dc_/brg_ read from the union's area-weighted centroid (equal-area
    # segments -> their midpoint (50, 0)).
    assert abs(rows["d0"]["center distance to st_peters"]
               - (40.0 ** 2 + 10.0 ** 2) ** 0.5) < 0.1, rows["d0"]
    hdr = out["results"]["r1"].refs[0]
    assert hdr["kind"] == "union", hdr
    assert hdr["members"] == ["road_86", "road_88"], hdr


def test_union_brg_gap_witness_is_the_argmin_members():
    # d1 sits just NORTH (+y) of SEG_B's wall: the local min-gap direction
    # reads ~0 (due north) from THAT member's witness pair -- not the
    # direction of the far union centroid at (50,0) (which reads ~east,
    # ~83 deg).
    cands = [{"fid": "d1", "xy": [90.0, 5.0], "class": "car"}]
    out = _union_run(cands)
    r = {x["fid"]: x for x in out["results"]["r1"]}["d1"]
    assert r["facing st_peters"] == "south", r
    assert 70.0 < r["bearing from st_peters"] < 100.0, r    # centroid bearing differs


def test_union_brg_gap_absent_on_contact_with_any_member():
    cands = [{"fid": "d2", "xy": [10.0, 0.0], "class": "car"}]  # inside SEG_A
    out = _union_run(cands)
    r = {x["fid"]: x for x in out["results"]["r1"]}["d2"]
    assert r["distance to st_peters"] == 0.0, r
    assert "facing st_peters" not in r, list(r)
    assert "bearing from st_peters" in r and "center distance to st_peters" in r, list(r)


def test_union_column_orders_in_emphasis():
    # The attached union column feeds score_by like any other -- the
    # already-attached short-circuit means auto-attach leaves it verbatim.
    cands = [
        {"fid": "far", "xy": [50.0, 40.0], "class": "car"},
        {"fid": "near", "xy": [90.0, 4.0], "class": "car"},
    ]
    out = _union_run(cands, extra_ops=[
        {"ref": "r2", "op": "score_by",
         "args": {"items": "r1",
                  "emphasis": [{"key": "distance to st_peters", "w": -1}]}},
    ])
    ranked = [r["fid"] for r in out["results"]["r2"]]
    assert ranked == ["near", "far"], ranked


def test_union_without_as_label_is_loud():
    try:
        run_program([{"fid": "d0", "xy": [0.0, 0.0], "class": "car"}],
                    [dict(_SEG_A), dict(_SEG_B)], {"ops": [
            {"ref": "r0", "op": "compare",
             "args": {"items": "cands",
                      "refs": [{"ref": ["road_86", "road_88"]}]}},
        ]})
        raise AssertionError("union ref without as:'label' did not error")
    except ProgramError as e:
        assert "as:'label'" in str(e), str(e)


def test_union_unknown_member_is_loud():
    try:
        run_program([{"fid": "d0", "xy": [0.0, 0.0], "class": "car"}],
                    [dict(_SEG_A)], {"ops": [
            {"ref": "r0", "op": "compare",
             "args": {"items": "cands",
                      "refs": [{"ref": ["road_86", "road_99"],
                                "as": "st_peters"}]}},
        ]})
        raise AssertionError("unknown union member did not error")
    except ProgramError as e:
        assert "road_99" in str(e), str(e)


def _main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL UNIVERSAL-RELATION TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
