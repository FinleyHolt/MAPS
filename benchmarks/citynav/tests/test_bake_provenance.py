"""Provenance travels with the tiles.

A city bake writes ``meta.json`` ``bake.mode`` + ``provenance`` (the cache's
name and sha256, its record count, the georef residuals it carries, bake time
and git sha), and never keeps the previous bake's claim (``_STALE_BAKE_FLAGS``).
"""

from __future__ import annotations

import hashlib
import json

from benchmarks.citynav.bake.city import bake_city


def test_a_city_bake_stamps_the_cache_it_read(tmp_path):
    block = "birmingham_block_0"
    (tmp_path / f"{block}_meta.json").write_text(json.dumps({
        "map_name": block,
        "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 40.0, "y_max": 40.0},
        "grid_shape": [20, 20],
        "cell_size_m": 2.0,
        "provenance": {"prior": "stale"},
    }))
    cache = tmp_path / "birmingham_osm_local.geojson"
    cache.write_text(json.dumps({
        "type": "FeatureCollection",
        "citynav_osm_cache": {"georef_provenance": {"passed": True}},
        "features": [{"type": "Feature", "id": "w1", "properties": {"building": "yes"},
                      "geometry": {"type": "Polygon",
                                   "coordinates": [[[2, 2], [10, 2], [10, 10], [2, 10], [2, 2]]]}}],
    }))
    bake_city("birmingham", cache, tmp_path)
    meta = json.loads((tmp_path / f"{block}_meta.json").read_text())
    assert meta["bake"]["mode"] == "osm_blocks"
    prov = meta["provenance"]
    assert prov["prior"] == "osm"
    assert prov["osm_cache"] == cache.name
    assert prov["osm_cache_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert prov["records"] == 1
    assert prov["georef"] == {"passed": True}
    assert prov["baked_at_utc"]
    assert "bake_git_sha" in prov
