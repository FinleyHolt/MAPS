"""A CityNav block is its raster: bounds and grid come from the imagery.

The bake needs no seed meta: every ``<city>_block_<n>.tif`` is a block, its
extent the bounds, on 2 m cells. A meta sidecar already in the output
directory is overwritten, never read; a block without a readable raster is
a named failure.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.citynav.bake.city import BakeError, bake_block
from maps.mission.tests.ortho_fixture import write_rasters

BLOCK = "birmingham_block_9"
POINT = ({"amenity": "bench", "osm_id": "n1"}, {"type": "Point", "coordinates": [810.0, 805.0]})


@pytest.fixture
def rasters(tmp_path):
    """The raster directory every bake here reads its block from."""
    return tmp_path / "rasters"


def _raster(rasters) -> None:
    """A 40 m x 20 m raster at (800, 800)."""
    write_rasters(
        rasters,
        BLOCK,
        np.full((200, 400, 3), 90, dtype=np.uint8),
        np.zeros((200, 400), dtype=np.float32),
        x_min=800.0,
        y_min=800.0,
    )


def _baked_meta(tiles, rasters) -> dict:
    bake_block(BLOCK, [POINT], tiles, rgbd_dir=rasters)
    return json.loads((tiles / f"{BLOCK}_meta.json").read_text())


def test_bounds_and_grid_come_from_the_raster_into_an_empty_tiles_dir(tmp_path, rasters):
    """Rows run north-to-south, so the grid is (y, x) = (10, 20); swapping them
    would transpose every non-square block."""
    _raster(rasters)
    meta = _baked_meta(tmp_path / "tiles", rasters)
    assert meta["bounds"] == {"x_min": 800.0, "y_min": 800.0, "x_max": 840.0, "y_max": 820.0}
    assert meta["grid_shape"] == [10, 20] and meta["cell_size_m"] == 2.0


def test_a_left_over_sidecar_is_not_read(tmp_path, rasters):
    _raster(rasters)
    (tmp_path / f"{BLOCK}_meta.json").write_text(
        json.dumps(
            {
                "cell_size_m": 5.0,
                "grid_shape": [60, 70],
                "bounds": {"x_min": 750.0, "y_min": 750.0, "x_max": 890.0, "y_max": 870.0},
            }
        )
    )
    meta = _baked_meta(tmp_path, rasters)
    assert meta["bounds"]["x_min"] == 800.0 and meta["grid_shape"] == [10, 20]


def test_a_block_without_a_raster_is_a_named_failure(tmp_path, rasters):
    with pytest.raises(BakeError, match=f"{BLOCK}.tif"):
        bake_block(BLOCK, [POINT], tmp_path, rgbd_dir=rasters)
