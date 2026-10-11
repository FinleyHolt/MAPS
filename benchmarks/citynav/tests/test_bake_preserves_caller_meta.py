"""A bake keeps the meta keys its caller wrote and re-asserts its own.

A top-level ``meta["georef"]`` is written by an upstream caller
(``benchmarks.calcity.data_prep.bake_tiles._write_meta``, which stamps the
AOI's UTM zone and local origin), never by the baker, so it reaches the tile a
global-coords caller reads. A previous bake's ``bake_counts`` and
``provenance`` never survive a re-bake.
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


def test_caller_meta_survives_and_stale_bake_flags_do_not(tmp_path):
    (tmp_path / "calcity_meta.json").write_text(json.dumps({
        "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 20.0, "y_max": 20.0},
        "grid_shape": [10, 10],
        "cell_size_m": 2.0,
        "coordinate_frame": {"x_axis": "east", "y_axis": "north", "origin": "southwest"},
        "georef": CALCITY_GEOREF,
        "bake_counts": {"named_kept": 999},
        "provenance": {"prior": "stale"},
    }))
    road = ({"name": "Road", "highway": "residential", "osm_id": "w1"},
            {"type": "LineString", "coordinates": [[1.0, 5.0], [19.0, 5.0]]})
    bake_block("calcity", [road], tmp_path, imagery=False)
    out = json.loads((tmp_path / "calcity_meta.json").read_text())
    assert out["georef"] == CALCITY_GEOREF, (
        "the caller's georef must reach the tile: tour_to_tasking --coords "
        "global and plot_ao_osm read crs_utm_epsg/origin_e/origin_n from it"
    )
    assert out["coordinate_frame"]["y_axis"] == "north"
    assert out["bake_counts"]["named_kept"] == 1, "re-asserted, not inherited"
    assert "provenance" not in out, "an osm_cache bake stamps no provenance"
    assert out["bake"]["mode"] == "osm_cache"
