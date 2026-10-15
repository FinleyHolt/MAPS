"""A bake's meta is the grid, then the caller's keys as given.

Cal City's ``georef`` (``benchmarks.calcity.data_prep.bake_tiles.bake_block``,
the AOI's UTM zone and local origin) reaches the tile a global-coords reader
(``benchmarks.flight.tile_georef``) reads; its ``coordinate_frame`` feeds the
y-north guard (``benchmarks.calcity.paths``). Nothing a previous bake left
survives.
"""

from __future__ import annotations

import json

from benchmarks.citynav.bake.city import bake_block
from benchmarks.citynav.eval import WorldBounds

CALCITY_GEOREF = {
    "crs_utm_epsg": 32611,
    "origin_e": 421234.5678,
    "origin_n": 3885432.1098,
}


def test_the_caller_frame_and_georef_follow_the_grid_and_nothing_else_does(tmp_path):
    grid = {"grid_shape": [10, 10], "cell_size_m": 2.0,
            "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 20.0, "y_max": 20.0}}
    frame = {"y_axis": "north"}
    (tmp_path / "calcity_meta.json").write_text(json.dumps({"provenance": {"prior": "stale"}}))
    road = ({"name": "Road", "highway": "residential", "osm_id": "w1"},
            {"type": "LineString", "coordinates": [[1.0, 5.0], [19.0, 5.0]]})
    bake_block("calcity", [road], tmp_path, bounds=WorldBounds(x_min=0.0, y_min=0.0, x_max=20.0, y_max=20.0),
               meta={"coordinate_frame": frame, "georef": CALCITY_GEOREF})
    out = (tmp_path / "calcity_meta.json").read_text()
    assert out == json.dumps({**grid, "coordinate_frame": frame, "georef": CALCITY_GEOREF}, indent=2)
