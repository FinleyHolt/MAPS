"""A bake's meta is the grid, its caller's frame and ``georef``, and its own provenance.

A top-level ``meta["georef"]`` is written by an upstream caller
(``benchmarks.calcity.data_prep.bake_tiles._write_meta``, which stamps the
AOI's UTM zone and local origin), never by the baker, so it reaches the tile a
global-coords reader (``benchmarks.flight.tile_georef``) reads; its
``coordinate_frame`` feeds the y-north guard (``benchmarks.calcity.paths``).
Every other key a caller or a previous bake left is dropped.
"""

from __future__ import annotations

import json

from benchmarks.citynav.bake.city import bake_block

CALCITY_GEOREF = {
    "crs_utm_epsg": 32611,
    "origin_e": 421234.5678,
    "origin_n": 3885432.1098,
    "runtime_transform": "x=easting-origin_e; y=northing-origin_n",
    "aoi_bbox_wgs84": [-118.010, 35.0988, -117.9385, 35.155],
}


def test_the_caller_frame_and_georef_survive_and_nothing_else_does(tmp_path):
    grid = {"grid_shape": [10, 10], "cell_size_m": 2.0,
            "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 20.0, "y_max": 20.0}}
    frame = {"x_axis": "east", "y_axis": "north", "origin": "southwest"}
    (tmp_path / "calcity_meta.json").write_text(json.dumps({
        **grid,
        "block": "calcity",
        "coordinate_frame": frame,
        "georef": CALCITY_GEOREF,
        "bake_counts": {"named_kept": 999},
        "provenance": {"prior": "stale"},
    }))
    road = ({"name": "Road", "highway": "residential", "osm_id": "w1"},
            {"type": "LineString", "coordinates": [[1.0, 5.0], [19.0, 5.0]]})
    bake_block("calcity", [road], tmp_path, imagery=False)
    out = json.loads((tmp_path / "calcity_meta.json").read_text())
    assert out == {**grid, "coordinate_frame": frame, "georef": CALCITY_GEOREF}
