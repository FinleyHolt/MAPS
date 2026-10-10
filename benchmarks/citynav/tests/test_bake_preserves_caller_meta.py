"""``_write_outputs`` drops only the flags this baker re-asserts.

Regression cover: ``georef`` sat in ``_STALE_BAKE_FLAGS`` while nothing in
this module ever wrote a top-level ``meta["georef"]`` — the only producer is
an upstream caller (``benchmarks.calcity.data_prep.bake_tiles._write_meta``,
which stamps the AOI's UTM zone and local origin). So the Cal City bake wrote
the block and ``bake_city_from_cache`` deleted it moments later, every time.
The strip landed 2026-07-15, the writer 2026-07-27: it never once survived,
and the georef a global-coords caller reads was absent on
any Cal City tile.
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


def test_every_stale_flag_is_one_this_baker_reasserts():
    """The rule that keeps georef from being re-added by reflex: a key belongs
    in the set only if some bake path writes it back. ``georef`` failed this —
    no ``flags`` dict in the module has ever carried it."""
    source = Path(__file__).resolve().parents[1].joinpath("bake", "city.py").read_text()
    # Every member must appear as a key in a flags/provenance literal, i.e.
    # somewhere this module assigns it, not merely in the strip set itself.
    for key in _STALE_BAKE_FLAGS:
        assert source.count(f'"{key}"') > 1, (
            f"{key!r} is stripped but never re-asserted by this baker — it can "
            "only belong to an upstream caller, and stripping deletes it"
        )
