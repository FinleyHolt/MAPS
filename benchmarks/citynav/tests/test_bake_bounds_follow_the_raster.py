"""A block's bounds are its imagery's bounds.

The meta sidecar box runs ~50 m past the raster on every side, where no
CityRefer object or CityNav target lies, so the bake derives bounds from the
raster and sizes the grid to match, falling back to the sidecar box (and
saying so) when no raster is readable.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import benchmarks.citynav.paths as citynav_paths
from benchmarks.citynav.bake.city import BakeError, bake_block
from maps.search.tests.ortho_fixture import write_rasters

BLOCK = "birmingham_block_9"
POINT = ({"amenity": "bench", "osm_id": "n1"}, {"type": "Point", "coordinates": [810.0, 805.0]})


@pytest.fixture
def tiles(tmp_path, monkeypatch):
    """A tiles dir whose sidecar claims a box 50 m larger than the imagery."""
    (tmp_path / f"{BLOCK}_meta.json").write_text(json.dumps({
        "block": BLOCK,
        "cell_size_m": 2.0,
        "grid_shape": [60, 70],
        "bounds": {"x_min": 750.0, "y_min": 750.0, "x_max": 890.0, "y_max": 870.0},
    }))
    monkeypatch.setattr(citynav_paths, "CITYNAV_NATIVE_RGBD_DIR", tmp_path / "rasters")
    return tmp_path


def _baked_meta(tiles) -> dict:
    bake_block(BLOCK, [POINT], tiles)
    return json.loads((tiles / f"{BLOCK}_meta.json").read_text())


def test_bounds_and_grid_come_from_the_raster(tiles):
    """A 40 m x 20 m raster at (800, 800): rows run north-to-south, so the grid
    is (y, x) = (10, 20); swapping them would transpose every non-square block."""
    write_rasters(tiles / "rasters", BLOCK, np.full((200, 400, 3), 90, dtype=np.uint8),
                  np.zeros((200, 400), dtype=np.float32), x_min=800.0, y_min=800.0)
    meta = _baked_meta(tiles)
    assert meta["bounds"] == {"x_min": 800.0, "y_min": 800.0, "x_max": 840.0, "y_max": 820.0}
    assert meta["grid_shape"] == [10, 20]
    assert meta["bounds_source"] == "block raster"
    assert meta["coverage_source"] == "ortho alpha"


def test_without_a_raster_the_sidecar_stands_and_says_so(tiles):
    meta = _baked_meta(tiles)
    assert meta["bounds"]["x_min"] == 750.0 and meta["bounds"]["x_max"] == 890.0
    assert meta["grid_shape"] == [60, 70]
    assert meta["bounds_source"] == "meta sidecar"
    assert meta["coverage_source"] == "bounds rectangle"


def test_a_missing_sidecar_is_a_named_failure(tmp_path):
    with pytest.raises(BakeError, match=f"{BLOCK}_meta.json"):
        bake_block(BLOCK, [POINT], tmp_path)
