"""A block's bounds are its imagery's bounds.

The block's meta sidecar box runs ~50 m past its raster on every side. OSM
covers the planet, so that ring would fill with features, but the point cloud,
the raster and the ortho all stop at the same line, and no upstream CityRefer
object (26,274 checked, 0 outside) or CityNav target (22,002 checked, 0
outside) lives past it.

So the bake derives bounds from the raster and sizes the grid to match. These
tests pin that, including the fallback when no raster is readable, which says
so when it falls back to the sidecar box.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.citynav.bake import city as bake_city


@pytest.fixture
def tiles(tmp_path):
    """A tiles dir whose sidecar claims a box 50 m larger than the imagery."""
    (tmp_path / "birmingham_block_9_meta.json").write_text(
        json.dumps(
            {
                "block": "birmingham_block_9",
                "cell_size_m": 2.0,
                "grid_shape": [250, 250],
                "bounds": {
                    "x_min": 750.0,
                    "y_min": 750.0,
                    "x_max": 1250.0,
                    "y_max": 1250.0,
                },
            }
        )
    )
    return tmp_path


def test_bounds_come_from_the_raster_not_the_sidecar(tiles, monkeypatch):
    monkeypatch.setattr(
        bake_city, "_raster_bounds", lambda block: (800.0, 800.0, 1200.0, 1200.0)
    )
    meta, bounds, grid_shape, cell = bake_city._load_city_meta(
        "birmingham_block_9", tiles
    )

    assert (bounds.x_min, bounds.y_min) == (800.0, 800.0)
    assert (bounds.x_max, bounds.y_max) == (1200.0, 1200.0)
    # Grid is sized to the imagery, not carried over from the wider box.
    assert grid_shape == (200, 200)
    assert cell == 2.0
    # The returned meta records what happened, so a later reader can tell.
    assert meta["bounds_source"] == "block raster"
    assert meta["bounds"]["x_min"] == 800.0
    assert meta["grid_shape"] == [200, 200]


def test_a_non_square_raster_sizes_rows_and_cols_independently(tiles, monkeypatch):
    """Rows run north-to-south, so grid_shape is (y, x). Swapping them silently
    transposes every block that is not square — and most are not."""
    monkeypatch.setattr(
        bake_city, "_raster_bounds", lambda block: (800.0, 800.0, 1200.0, 1000.0)
    )
    _meta, bounds, grid_shape, _cell = bake_city._load_city_meta(
        "birmingham_block_9", tiles
    )
    assert bounds.x_max - bounds.x_min == 400.0  # 200 cols
    assert bounds.y_max - bounds.y_min == 200.0  # 100 rows
    assert grid_shape == (100, 200)


def test_without_a_raster_the_sidecar_stands_and_says_so(tiles, monkeypatch):
    """A missing raster falls back rather than failing the bake, but the meta
    must not claim the bounds came from imagery."""
    monkeypatch.setattr(bake_city, "_raster_bounds", lambda block: None)
    meta, bounds, grid_shape, _cell = bake_city._load_city_meta(
        "birmingham_block_9", tiles
    )
    assert (bounds.x_min, bounds.x_max) == (750.0, 1250.0)
    assert grid_shape == (250, 250)
    assert meta["bounds_source"] == "meta sidecar"


def test_a_missing_sidecar_is_a_named_failure(tmp_path):
    with pytest.raises(bake_city.BakeError) as exc:
        bake_city._load_city_meta("birmingham_block_9", tmp_path)
    assert "birmingham_block_9_meta.json" in str(exc.value)
