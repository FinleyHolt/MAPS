"""Provenance travels with the tiles.

A city bake writes ``meta.json`` ``provenance`` (the cache's name and sha256,
its record count, the georef residuals it carries, bake time and git sha) for
every block raster, and never keeps the previous bake's claim.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

import benchmarks.citynav.paths as citynav_paths
from benchmarks.citynav.bake.city import bake_city
from maps.search.tests.ortho_fixture import write_rasters


def test_a_city_bake_stamps_the_cache_it_read(tmp_path, monkeypatch):
    block = "birmingham_block_0"
    write_rasters(tmp_path / "rasters", block, np.full((400, 400, 3), 90, dtype=np.uint8),
                  np.zeros((400, 400), dtype=np.float32))
    monkeypatch.setattr(citynav_paths, "CITYNAV_NATIVE_RGBD_DIR", tmp_path / "rasters")
    (tmp_path / f"{block}_meta.json").write_text(json.dumps({"provenance": {"prior": "stale"}}))
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
    prov = meta["provenance"]
    assert prov["prior"] == "osm"
    assert prov["osm_cache"] == cache.name
    assert prov["osm_cache_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert prov["records"] == 1
    assert prov["georef"] == {"passed": True}
    assert prov["baked_at_utc"]
    assert "bake_git_sha" in prov
    assert len(prov) == 7, sorted(prov)
