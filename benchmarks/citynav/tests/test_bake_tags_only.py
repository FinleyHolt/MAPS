"""Tags-only bake: no classifier, no drop-gates, provenance fids, kind census.

Pins the bake contract: a prior is { mask (uniform realizer), ALL raw OSM
tags verbatim, name when present, osm_id }. The ONLY exclusion is
no-geometry-in-window; the graph builder mints ``prior_<i>`` fids
(provenance_index, mirroring ``detected_<i>``); the kind census groups by
mask kind + top tag key=value counts.
"""
from __future__ import annotations

import json

from benchmarks.citynav.bake.city import bake_block
from maps.sws.graph.io import load_graph


def _poly(x0, y0, x1, y1):
    return {"type": "Polygon",
            "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


_RECORDS = [
    # a tag combination NO class table ever spoke for — kept regardless
    ({"power": "generator", "osm_id": "w1"}, _poly(2, 2, 10, 10)),
    ({"amenity": "parking", "osm_id": "w2"}, _poly(12, 2, 20, 10)),
    ({"name": "Elm Street", "highway": "residential", "osm_id": "w3"},
     {"type": "LineString", "coordinates": [[0, 20], [40, 20]]}),
    ({"tourism": "artwork", "osm_id": "n1"},
     {"type": "Point", "coordinates": [30, 30]}),
    # outside the window: rasterizes to no cell — the ONE exclusion
    ({"building": "yes", "osm_id": "w4"}, _poly(200, 200, 210, 210)),
]


def _bake(tmp_path):
    """A 40 m x 40 m map of 2 m cells; returns its prior rows, meta and graph."""
    (tmp_path / "toy_meta.json").write_text(json.dumps({
        "cell_size_m": 2.0, "grid_shape": [20, 20],
        "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 40.0, "y_max": 40.0},
    }))
    bake_block("toy", list(_RECORDS), tmp_path, imagery=False)
    rows = json.loads((tmp_path / "toy_features.json").read_text())["prior"]
    meta = json.loads((tmp_path / "toy_meta.json").read_text())
    return rows, meta, load_graph(str(tmp_path / "toy"))


def test_every_in_window_geometry_is_kept_with_its_tags_and_nothing_minted(tmp_path):
    rows, _meta, graph = _bake(tmp_path)
    # 4 in-window records kept — including the classifier-orphan tags —
    # exactly 1 excluded, and ONLY for no-geometry-in-window.
    by_id = {e["osm_id"]: e for e in rows}
    assert set(by_id) == {"w1", "w2", "w3", "n1"}, by_id
    gen = by_id["w1"]
    assert gen["power"] == "generator"     # raw tag verbatim
    assert "name" not in gen               # no synthesized name
    assert "prior_role" not in gen         # nothing minted
    assert gen["cells"]                    # the realised mask
    assert by_id["w3"]["name"] == "Elm Street"   # name when present
    # fids are prior_<index>, mirroring detected_<index>
    assert set(graph.nodes) == {f"prior_{i}" for i in range(4)}, sorted(graph.nodes)


def test_a_polygon_ring_is_not_decimated():
    """The OSM ring IS the footprint, so it rides whole.

    _mask_fields prefers this contour over the cell boundary, so it is the
    footprint for every areal prior; a decimated ring is a wrong
    footprint_area_m2, which the model scores and filters on.
    """
    import math

    from benchmarks.citynav.bake.city import _world_contour_from_geometry

    # A 64-gon: every vertex carries shape, and a stride would drop 3 of 4.
    ring = [
        [round(100.0 * math.cos(t), 3), round(100.0 * math.sin(t), 3)]
        for t in [i * 2 * math.pi / 64 for i in range(64)]
    ]
    out = _world_contour_from_geometry({"type": "Polygon", "coordinates": [ring]})
    assert out is not None
    # 64 vertices plus the closing one that says the chain encloses.
    assert len(out) == 65, f"ring decimated to {len(out)} vertices"
    assert out[0] == out[-1], (out[0], out[-1])
    out = out[:-1]

    def area(r):
        return abs(
            sum(
                r[i][0] * r[(i + 1) % len(r)][1] - r[(i + 1) % len(r)][0] * r[i][1]
                for i in range(len(r))
            )
        ) / 2.0

    # Area survives; a 16-vertex stride of this ring loses several percent.
    assert abs(area(out) - area(ring)) / area(ring) < 0.001


def test_the_bake_keeps_whatever_geometry_it_is_given():
    """No kind is worth less than another. A polygon's ring closes, a way's
    vertices ride open, a node is its one vertex -- and nothing here asks
    what the feature IS."""
    from benchmarks.citynav.bake.city import _world_contour_from_geometry as wc

    way = [[0.0, 0.0], [10.0, 0.0], [10.0, 8.0]]
    assert wc({"type": "LineString", "coordinates": way}) == way
    assert wc({"type": "Point", "coordinates": [3.0, 4.0]}) == [[3.0, 4.0]]

    square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
    assert wc({"type": "Polygon", "coordinates": [square]}) == square

    # A way that returns to its start (an OSM roundabout) IS a ring, by the
    # same rule and with no special case for it.
    loop = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 0.0]]
    assert wc({"type": "LineString", "coordinates": loop}) == loop

    # Multi-part: the largest part by extent, the same rule for either kind.
    assert wc({"type": "MultiLineString",
               "coordinates": [[[0.0, 0.0], [1.0, 0.0]], way]}) == way
    big = [[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0], [0.0, 0.0]]
    assert wc({"type": "MultiPolygon",
               "coordinates": [[square], [big]]}) == big

    # Degenerate: a chain that rounds onto one place IS that place, and a
    # ring that rounds below three distinct vertices stops being a ring.
    assert wc({"type": "LineString",
               "coordinates": [[1.0, 1.0], [1.001, 1.0]]}) == [[1.0, 1.0]]
    assert wc({"type": "Polygon",
               "coordinates": [[[0.0, 0.0], [0.001, 0.0], [0.0, 0.001],
                                [0.0, 0.0]]]}) is None


def test_a_point_just_outside_the_window_is_dropped(tmp_path):
    """A point 1 m west or south of the map lies in no cell; it is not
    pulled onto the edge column or row."""
    (tmp_path / "toy_meta.json").write_text(json.dumps({
        "cell_size_m": 2.0, "grid_shape": [20, 20],
        "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 40.0, "y_max": 40.0},
    }))
    bake_block("toy", [
        ({"tourism": "artwork", "osm_id": "n1"}, {"type": "Point", "coordinates": [30, 30]}),
        ({"tourism": "artwork", "osm_id": "w"}, {"type": "Point", "coordinates": [-1, 30]}),
        ({"tourism": "artwork", "osm_id": "s"}, {"type": "Point", "coordinates": [30, -1]}),
    ], tmp_path, imagery=False)
    rows = json.loads((tmp_path / "toy_features.json").read_text())["prior"]
    assert {e["osm_id"] for e in rows} == {"n1"}, rows
