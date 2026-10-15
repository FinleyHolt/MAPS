"""A row is ONE OSM way, and a name is the SET of its rows."""

import numpy as np

from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.geom_program import run_program
from maps.search.analyze_scene.surface import build_surface_from_graph
from maps.sws.graph.model import MapObject


def _way(name, cells, tags=None):
    cells = np.asarray(cells, dtype=int).reshape(-1, 2)
    return MapObject(class_name="", name=name, tags=tags or {},
                     centroid=(float(cells[:, 0].mean()), float(cells[:, 1].mean())),
                     cells=cells)


def _road(name, c0, c1, row=10):
    return _way(name, [[row, c] for c in range(c0, c1 + 1)])


def test_each_osm_way_keeps_its_own_row():
    """A name resolves to a SET, not to one folded row."""
    ways = [("prior_9", _road("Lime Avenue", 20, 29)),
            ("prior_3", _road("Lime Avenue", 0, 9)),
            ("prior_7", _road("Lime Avenue", 10, 19)),
            ("prior_1", _way("Oak Street", [[0, 0]]))]
    _c, anchors = build_surface_from_graph(ways, (0.0, 0.0), 2.0)
    assert len(anchors) == 4, [a["fid"] for a in anchors]
    lime = [a for a in anchors if a["name"] == "Lime Avenue"]
    assert len(lime) == 3, [a["fid"] for a in lime]
    # Each row is one way's worth of ground, never the union.
    assert sorted(len(a["cells"]) for a in lime) == [10, 10, 10]


def test_a_rows_cells_are_its_own_ways_cells():
    ways = [("prior_1", _road("Lime Avenue", 0, 9)),
            ("prior_2", _road("Lime Avenue", 10, 19)),
            ("prior_3", _road("Lime Avenue", 400, 409))]
    _c, anchors = build_surface_from_graph(ways, (0.0, 0.0), 2.0)
    by_fid = dict(ways)
    assert len(anchors) == 3
    for row in anchors:
        node = by_fid[row["fid"]]
        assert len(row["cells"]) == len(node.cells), row["fid"]
        assert {tuple(c) for c in row["cells"]} == \
            {tuple(c) for c in node.cells.tolist()}, row["fid"]


def test_every_row_is_one_place_and_may_claim_a_centre():
    """One way is one place, so the centre columns read on every named row."""
    _c, anchors = build_surface_from_graph(
        [("prior_1", _road("Lime Avenue", 0, 9)),
         ("prior_2", _road("Lime Avenue", 400, 409))], (0.0, 0.0), 2.0)
    lime = [a for a in anchors if a["name"] == "Lime Avenue"]
    assert len(lime) == 2, [a["fid"] for a in lime]
    for row in lime:
        assert G.one_place(row["mask"]), row["mask"]
        assert row.get("mask_centroid") is not None, row
        off, _a, _b = G.gap_witness(
            G.mask_geom(row["mask"]),
            G.mask_geom([[list(row["mask_centroid"])]]))
        assert off <= 2.0, (row["fid"], row["mask_centroid"], off)
    both = [ch for r in lime for ch in r["mask"]]
    ghost = G.loops_area_centroid(both)
    off, _a, _b = G.gap_witness(G.mask_geom(both), G.mask_geom([[list(ghost)]]))
    assert off > 300.0, (ghost, off)


def test_a_row_that_is_STILL_several_places_orients_no_side():
    """The ``mask_center`` law, on the one shape that can still reach it: a
    reference that is several places serves its gap and no centre column."""
    speckled = _way(None, [[10, c] for c in range(0, 10)]
                    + [[10, c] for c in range(400, 410)])
    _c, anchors = build_surface_from_graph([("prior_1", speckled)], (0.0, 0.0), 2.0)
    assert not G.one_place(anchors[0]["mask"]), anchors[0]["mask"]
    assert anchors[0].get("mask_centroid") is None, anchors[0]

    out = run_program([{"fid": "d0", "xy": [83.0, 27.0], "class": "car"}],
                      [dict(a) for a in anchors],
                      {"ops": [{"ref": "r0", "op": "compare",
                                "args": {"items": ["d0"], "refs": ["prior_1"]}}]})
    row = out["results"]["r0"][0]
    assert "distance to prior_1" in row, row
    assert "center distance to prior_1" not in row, row
    assert "bearing from prior_1" not in row, row


def test_the_surface_carries_cells_and_a_name_repeats_across_its_ways():
    """Duplicate names are the POINT: they are what ``.exact`` collects."""
    ways = [(f"prior_{i}", _road("Lime Avenue", i * 10, i * 10 + 9))
            for i in range(4)]
    ways.append(("prior_99", _way("Oak Street", [[40, 40], [40, 41]])))
    _cands, anchors = build_surface_from_graph(ways, (0.0, 0.0), 2.0)
    assert len(anchors) == 5
    lime = [a for a in anchors if a["name"] == "Lime Avenue"]
    assert len(lime) == 4, [a["fid"] for a in lime]
    assert all(len(a["cells"]) == 10 for a in lime)
    # The road is the SET: its ways together carry all 40 cells, once each.
    assert len({tuple(c) for a in lime for c in a["cells"]}) == 40
