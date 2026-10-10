"""Tags-only bake: no classifier, no drop-gates, provenance fids, kind census.

Pins the bake contract: a prior is { mask (uniform realizer), ALL raw OSM
tags verbatim, name when present, osm_id }. The ONLY exclusion is
no-geometry-in-window; the graph builder mints ``prior_<i>`` fids
(provenance_index, mirroring ``detected_<i>``); the kind census groups by
mask kind + top tag key=value counts.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)

from benchmarks.citynav.bake.city import (  # noqa: E402
    PRIOR_KEY,
    _features_from_records,
    _overpass_all_elements_query,
)
from benchmarks.citynav.eval import WorldBounds  # noqa: E402
from maps.sws.graph import build_object_graph  # noqa: E402

_BOUNDS = WorldBounds(x_min=0.0, y_min=0.0, x_max=40.0, y_max=40.0)
_GRID = (20, 20)  # 2 m cells


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


def _bake():
    counts: dict = {}
    features = _features_from_records(
        list(_RECORDS),
        bounds=_BOUNDS,
        grid_shape=_GRID,
        cell_size_m=2.0,
        line_buffer_m=1.0,
        bake_counts=counts,
    )
    return features, counts


def test_every_in_window_geometry_is_kept_no_class_gate():
    features, _counts = _bake()
    entries = features[PRIOR_KEY]
    # 4 in-window records kept — including the classifier-orphan tags —
    # exactly 1 excluded, and ONLY for no-geometry-in-window.
    assert len(entries) == 4, [e.get("osm_id") for e in entries]
    ids = {e["osm_id"] for e in entries}
    assert ids == {"w1", "w2", "w3", "n1"}, ids


def test_entries_are_mask_tags_name_osmid_nothing_minted():
    features, _counts = _bake()
    by_id = {e["osm_id"]: e for e in features[PRIOR_KEY]}
    gen = by_id["w1"]
    assert gen["power"] == "generator"     # raw tag verbatim
    assert "name" not in gen               # no synthesized name
    assert "prior_role" not in gen         # nothing minted
    assert gen["cells"]                    # the realised mask
    named = by_id["w3"]
    assert named["name"] == "Elm Street"   # name when present


def test_fids_are_prior_index_mirroring_detected():
    features, _counts = _bake()
    graph = build_object_graph(features, cell_size_m=2.0)
    fids = sorted(graph.nodes)
    assert all(f.startswith("prior_") for f in fids), fids
    assert set(fids) == {f"prior_{i}" for i in range(4)}, fids


def test_kind_census_groups_by_mask_kind_and_tag_counts():
    _features, counts = _bake()
    census = counts["kind_census"]
    assert census["mask_kinds"] == {"area": 2, "node": 1, "way": 1}, census
    top = census["top_tags"]
    assert top["amenity=parking"] == 1, top
    assert top["power=generator"] == 1, top
    assert top["highway=residential"] == 1, top


def test_fetch_query_is_unrestricted_nwr_no_tag_allowlist():
    # The fetch requests EVERY element in the window: one bare ``nwr`` bbox
    # clause with recursion, and NO tag filter of any kind — no quoted tag
    # keys, no key=value clauses. The bake's only exclusion is geometric.
    q = _overpass_all_elements_query([0.1052, 52.193, 0.1435, 52.2188])
    assert "nwr(52.193000,0.105200,52.218800,0.143500);" in q
    assert "(._;>;);out;" in q
    # no tag filter syntax anywhere: quoted keys / regex filters / k=v
    assert '"' not in q and "'" not in q and "~" not in q and "=" not in q
    # exactly the settings brackets, no ["key"] filter brackets
    assert q.startswith("[out:xml][timeout:")
    assert q.count("[") == 2, q


def test_a_polygon_ring_is_not_decimated():
    """The OSM ring IS the footprint, so it rides whole.

    _mask_fields prefers this contour over the cell boundary, so it is the
    footprint for every areal prior. Measured on the Cal City bundle, a
    16-vertex uniform stride would cost the 2.3% of rings past 16 vertices a
    median 13.4% of their area, at worst 99.1% -- a wrong footprint_area_m2,
    which the model scores and filters on.
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
    what the feature IS. The rule this replaced kept the ring and returned
    None for the other two, which is why 0 of 1,739 calcity road nodes
    carried exact vertices while 3,413 of 3,414 buildings did."""
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
