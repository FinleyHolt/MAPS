"""A row is ONE OSM way, and a name is the SET of its rows.

The property these tests exist for: a fid means the same thing on the
surface and in the runtime graph, so an op handed a row walks the ground
that row describes. A folded row broke it -- it carried the union of a
name's ways while its fid still addressed one of them, and ``explore``
resolves fids against the graph.

What the fold bought and this gives up: ONE served distance per name
whatever way the probe is nearest. The agent reassembles it from the set
(``find(name).exact``), which is also what it explores.
"""

import numpy as np

from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.geom_program import (
    ProgramError, build_surface, run_program,
)


def _served_gap(rows, name, probe_xy):
    """The distance the surface actually serves, from the EXACT mask loops.

    Not a cell-raster measurement: this is `gap_witness` over the same mask
    realisation the compare op resolves, so the tests pin the served number.
    """
    row = next(r for r in rows if r["name"] == name)
    probe = [[[probe_xy[0] - 0.5, probe_xy[1] - 0.5],
              [probe_xy[0] + 0.5, probe_xy[1] - 0.5],
              [probe_xy[0] + 0.5, probe_xy[1] + 0.5],
              [probe_xy[0] - 0.5, probe_xy[1] + 0.5]]]
    return G.gap_witness(G.mask_geom(row["mask"]), G.mask_geom(probe))[0]


class _Way:
    """The bit of MapObject the surface builder reads."""

    def __init__(self, name, cells, tags=None):
        self.name = name
        self.cells = np.asarray(cells, dtype=int).reshape(-1, 2)
        self.tags = tags or {}
        self.centroid = (float(self.cells[:, 0].mean()),
                         float(self.cells[:, 1].mean()))
        self.world_xy = None
        self.centerline = None
        self.class_name = ""
        self.world_contour = None
        self.cell_count = len(self.cells)


def _road(name, c0, c1, row=10):
    return _Way(name, [[row, c] for c in range(c0, c1 + 1)])


def test_each_osm_way_keeps_its_own_row():
    """A name resolves to a SET, not to one folded row.

    The ways of a road are its rows; ``find("Lime Avenue").exact`` is the
    road, and ``explore`` covers the set in one pass. Folding them cost the
    fid its meaning: a folded row carried the union of ten ways' cells while
    its fid still addressed ONE way in the runtime graph, so
    ``SearchLoop.node_coverage_stops`` walked that member alone and an
    ``explore`` of 3.8 km of Poppy Boulevard flew one stop and reported the
    road covered (calcity::200#198; Neuralia Road, 64 ways, walked 4 cells).
    """
    ways = [("prior_9", _road("Lime Avenue", 20, 29)),
            ("prior_3", _road("Lime Avenue", 0, 9)),
            ("prior_7", _road("Lime Avenue", 10, 19)),
            ("prior_1", _Way("Oak Street", [[0, 0]]))]
    _c, anchors = build_surface([], ways, (0.0, 0.0), 2.0)
    assert len(anchors) == 4, [a["fid"] for a in anchors]
    lime = [a for a in anchors if a["name"] == "Lime Avenue"]
    assert len(lime) == 3, [a["fid"] for a in lime]
    # Each row is one way's worth of ground, never the union.
    assert sorted(len(a["cells"]) for a in lime) == [10, 10, 10]


def test_a_rows_cells_are_its_own_ways_cells():
    """THE contract the fold broke: a fid means the same thing everywhere.

    ``explore`` resolves a row's fid against the runtime graph, so a row
    whose cells describe more ground than the node it names sends the walk
    over the wrong footprint. Asserted per row, so no row may quietly
    describe a neighbour's ground.
    """
    ways = [("prior_1", _road("Lime Avenue", 0, 9)),
            ("prior_2", _road("Lime Avenue", 10, 19)),
            ("prior_3", _road("Lime Avenue", 400, 409))]
    _c, anchors = build_surface([], ways, (0.0, 0.0), 2.0)
    by_fid = dict(ways)
    assert len(anchors) == 3
    for row in anchors:
        node = by_fid[row["fid"]]
        assert len(row["cells"]) == len(node.cells), row["fid"]
        assert {tuple(c) for c in row["cells"]} == \
            {tuple(c) for c in node.cells.tolist()}, row["fid"]


def test_every_row_is_one_place_and_may_claim_a_centre():
    """One way is one place, so the centre columns read on every named row.

    ``_mask_center`` refuses a row that is several places. Rows are ways
    now, so a NAME never lands in that refusal — two ways 869 m apart under
    one name are two rows, each centred on itself.
    """
    _c, anchors = build_surface(
        [], [("prior_1", _road("Lime Avenue", 0, 9)),
             ("prior_2", _road("Lime Avenue", 400, 409))], (0.0, 0.0), 2.0)
    lime = [a for a in anchors if a["name"] == "Lime Avenue"]
    assert len(lime) == 2, [a["fid"] for a in lime]
    for row in lime:
        assert G.chain_components(row["mask"]) == 1, row["mask"]
        assert row.get("mask_centroid") is not None, row
        off, _a, _b = G.gap_witness(
            G.mask_geom(row["mask"]),
            G.mask_geom([[list(row["mask_centroid"])]]))
        assert off <= 2.0, (row["fid"], row["mask_centroid"], off)
    # VACUITY FLOOR: a centre over BOTH places would be ~400 m from each,
    # which is the point a folded row used to serve.
    both = [ch for r in lime for ch in r["mask"]]
    ghost = G.loops_area_centroid(both)
    off, _a, _b = G.gap_witness(G.mask_geom(both), G.mask_geom([[list(ghost)]]))
    assert off > 300.0, (ghost, off)


def test_a_row_that_is_STILL_several_places_orients_no_side():
    """The ``_mask_center`` law, on the one shape that can still reach it.

    A row is one way, so a name never lands here — but a single way can
    still be several places: a mask the segmenter speckled, or a road the
    tile boundary cut into two in-tile pieces. The refusal has to stay
    reachable or it rots into a branch nothing ever turns over.
    """
    speckled = _Way(None, [[10, c] for c in range(0, 10)]
                    + [[10, c] for c in range(400, 410)])
    _c, anchors = build_surface([], [("prior_1", speckled)], (0.0, 0.0), 2.0)
    assert G.chain_components(anchors[0]["mask"]) == 2, anchors[0]["mask"]
    assert anchors[0].get("mask_centroid") is None, anchors[0]

    try:
        run_program([{"fid": "d0", "xy": [83.0, 27.0], "class": "car"}],
                    [dict(a) for a in anchors],
                    {"ops": [{"ref": "r0", "op": "compare",
                              "args": {"items": [{"fid": "d0",
                                                  "xy": [83.0, 27.0],
                                                  "class": "car"}],
                                       "frame": {"reference": "prior_1",
                                                 "side": "left"}}}]})
        raise AssertionError("a two-place reference oriented a side")
    except ProgramError as e:
        assert "several disjoint places" in str(e), str(e)


def test_the_surface_carries_cells_and_a_name_repeats_across_its_ways():
    """Duplicate names are the POINT: they are what ``.exact`` collects."""
    ways = [(f"prior_{i}", _road("Lime Avenue", i * 10, i * 10 + 9))
            for i in range(4)]
    ways.append(("prior_99", _Way("Oak Street", [[40, 40], [40, 41]])))
    _cands, anchors = build_surface([], ways, (0.0, 0.0), 2.0)
    assert len(anchors) == 5
    lime = [a for a in anchors if a["name"] == "Lime Avenue"]
    assert len(lime) == 4, [a["fid"] for a in lime]
    assert all(len(a["cells"]) == 10 for a in lime)
    # The road is the SET: its ways together carry all 40 cells, once each.
    assert len({tuple(c) for a in lime for c in a["cells"]}) == 40
