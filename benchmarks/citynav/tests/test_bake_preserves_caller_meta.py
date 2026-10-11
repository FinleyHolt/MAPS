"""``_write_outputs`` drops only the flags this baker re-asserts.

A top-level ``meta["georef"]`` is written by an upstream caller
(``benchmarks.calcity.data_prep.bake_tiles._write_meta``, which stamps the
AOI's UTM zone and local origin), never by this module, so it passes through
``bake_city_from_cache`` to the tile a global-coords caller reads.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.citynav.bake.city import _STALE_BAKE_FLAGS, _write_outputs

CALCITY_GEOREF = {
    "crs_utm_epsg": 32611,
    "origin_e": 421234.5678,
    "origin_n": 3885432.1098,
    "runtime_transform": "x=easting-origin_e; y=northing-origin_n",
    "aoi_bbox_wgs84": [-118.010, 35.0988, -117.9385, 35.155],
}


def _bake(tmp_path: Path, meta: dict) -> dict:
    _write_outputs(
        "calcity",
        tmp_path,
        meta,
        {"prior": [{"name": "Road", "object_type": "TrafficRoad", "cells": [[0, 0]]}]},
        (10, 10),
        2.0,
        source_mode="osm_blocks",
        flags={"osm_only": True, "bake_counts": {"named_kept": 1}},
    )
    return json.loads((tmp_path / "calcity_meta.json").read_text())


def test_caller_georef_survives_the_bake(tmp_path):
    out = _bake(
        tmp_path,
        {
            "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 20.0, "y_max": 20.0},
            "cell_size_m": 2.0,
            "coordinate_frame": {
                "x_axis": "east", "y_axis": "north", "origin": "southwest",
            },
            "georef": CALCITY_GEOREF,
        },
    )
    assert out["georef"] == CALCITY_GEOREF, (
        "the caller's georef must reach the tile — tour_to_tasking --coords "
        "global and plot_ao_osm read crs_utm_epsg/origin_e/origin_n from it"
    )
    # It rides beside the other facts the same caller stamped.
    assert out["coordinate_frame"]["y_axis"] == "north"


def test_a_stale_flag_is_still_stripped(tmp_path):
    """Vacuity floor. Emptying _STALE_BAKE_FLAGS would also make the test
    above pass, and would silently re-inherit provenance across a
    re-bake — the thing the set exists to prevent."""
    out = _bake(
        tmp_path,
        {
            "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 20.0, "y_max": 20.0},
            "cell_size_m": 2.0,
            "bake_counts": {"named_kept": 999},
            "provenance": {"prior": "stale"},
        },
    )
    assert out["bake_counts"] == {"named_kept": 1}, "re-asserted, not inherited"
    # provenance is re-asserted per bake, never inherited.
    assert out.get("provenance", {}).get("prior") != "stale"
    assert _STALE_BAKE_FLAGS, "the stale-flag set must not be empty"
