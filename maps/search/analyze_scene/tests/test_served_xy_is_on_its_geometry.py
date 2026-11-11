"""A row's served ``xy`` is the ``go_to``/``explore`` FLIGHT TARGET, so it is a place."""
from __future__ import annotations

import math

from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.surface import build_surface_from_graph, representative_point
from maps.world_state.graph import MapObject


def _node(contour, name=None, world_xy=None):
    """A map node with its exact geometry and the coarse 2 m mesh it
    rasterizes to."""
    contour = [(float(x), float(y)) for x, y in contour]
    cells = _cells_of(contour)
    return MapObject(
        class_name="", name=name,
        tags={"highway": "residential"} if name else {"building": "yes"},
        centroid=tuple(cells[0]), cells=cells, world_contour=contour,
        world_xy=world_xy)


def _cells_of(contour):
    import numpy as np
    return np.asarray([[int(y // 2), int(x // 2)] for x, y in contour],
                      dtype=int)


# A way bent around three sides of a block. It states no closure, so it is a POLYLINE
# and not a filled ring: its vertex mean is the middle of the block, which is the one
# place the way is not.
_BENT_WAY = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
# An ordinary building: convex, so the mean IS on it and nothing should move
# far.
_BUILDING = [(200.0, 200.0), (220.0, 200.0), (220.0, 214.0), (200.0, 214.0),
             (200.0, 200.0)]
# One NAME, two places 800 m apart -- the mean is 400 m from both.
_FAR_A = [(0.0, 500.0), (40.0, 500.0)]
_FAR_B = [(800.0, 500.0), (840.0, 500.0)]


def _surface():
    nodes = [("p_bent", _node(_BENT_WAY)),
             ("p_bldg", _node(_BUILDING)),
             ("p_far_a", _node(_FAR_A, name="Raymond Avenue")),
             ("p_far_b", _node(_FAR_B, name="Raymond Avenue"))]
    _cands, anchors = build_surface_from_graph(nodes, (0.0, 0.0), 2.0)
    return anchors


def _gap(row) -> float:
    """Metres from the row's served point to its own geometry. 0 = on it."""
    import numpy as np
    chains = [[[float(p[0]), float(p[1])] for p in ch]
              for ch in (row.get("mask") or []) if ch]
    geom = G.mask_geom(chains)
    xy = [float(row["xy"][0]), float(row["xy"][1])]
    if len(geom["CA"]) and bool(G.mask_contains(geom, [xy])[0]):
        return 0.0
    A, B = (geom["A"], geom["B"]) if len(geom["A"]) else (geom["V"], geom["V"])
    d, _f = G._pt_to_segs_grid(np.asarray([xy], dtype=float), A, B)
    return float(d.min())


def test_every_served_point_is_on_its_own_geometry():
    rows = _surface()
    # FOUR rows: the two far ways share a name and are still two rows. A
    # name folds on PROXIMITY as well, so two places under one name are two
    # places -- and each then has a point on itself rather than one row
    # having a point on whichever piece happened to be longest.
    assert len(rows) == 4, [r["fid"] for r in rows]
    for r in rows:
        assert _gap(r) <= 0.01, (r["fid"], r["xy"], _gap(r))


def test_a_measured_position_wins_over_a_derived_one():
    """A detection's ``world_xy`` is where the tracker put the object, sub-cell."""
    det = ("d_car", _node([(10.0, 10.0), (14.0, 10.0), (14.0, 12.0),
                           (10.0, 12.0), (10.0, 10.0)],
                          world_xy=(11.37, 10.94)))
    det[1].provenance = "detected"
    cands, _anchors = build_surface_from_graph([det], (0.0, 0.0), 2.0)
    assert cands[0]["xy"] == [11.37, 10.94], cands[0]["xy"]


def test_a_bent_way_takes_its_arc_midpoint_and_not_its_middle():
    p = representative_point([[list(v) for v in _BENT_WAY]])
    # 300 m of way; the arc midpoint is halfway UP the far side.
    assert math.isclose(p[0], 100.0, abs_tol=0.01), p
    assert math.isclose(p[1], 50.0, abs_tol=0.01), p


def test_a_convex_footprint_keeps_its_area_centroid():
    p = representative_point([[list(v) for v in _BUILDING]])
    assert math.isclose(p[0], 210.0, abs_tol=0.01), p
    assert math.isclose(p[1], 207.0, abs_tol=0.01), p
