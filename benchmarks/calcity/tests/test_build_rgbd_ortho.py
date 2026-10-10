"""The rgbd pair is built from the block's own map: the raster copied beside a
flat height plane on its grid, the pair the ortho loader streams."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from benchmarks.calcity.data_prep.aoi import GROUND_LEVEL_M, build_local_frame
from benchmarks.calcity.data_prep.build_rgbd_ortho import build_block
from benchmarks.calcity.data_prep.bundle import grid_shape
from maps.perception.ortho import camera_model as cm

# An 82 x 44 m patch of Cal City: ~270 x 145 px at 0.3 m.
BBOX = (-117.9700, 35.1200, -117.9691, 35.1204)
PX = 0.3


def _raster(path: Path, px: float = PX, seed: int = 0) -> np.ndarray:
    """A block raster over BBOX at ``px``: noise pixels in the frame's UTM georef."""
    lf = build_local_frame(BBOX)
    H, W = grid_shape(lf, px)
    rgb = np.random.default_rng(seed).integers(1, 256, (H, W, 3), dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8", crs=lf.crs_utm,
                       transform=from_origin(lf.origin_e, lf.origin_n + lf.height_m, px, px), compress="deflate") as dst:
        for b in range(3):
            dst.write(rgb[..., b], b + 1)
    return rgb


def test_build_block_writes_the_pair_the_loader_reads_at_the_rasters_cell(tmp_path: Path):
    (tmp_path / "survey").mkdir()
    rgb = _raster(tmp_path / "survey" / "calcity_rgb.tif")
    lf = build_local_frame(BBOX)
    root = tmp_path / "root"
    tif, rgb_tif = build_block(root, "patch", BBOX, tmp_path / "survey" / "calcity_rgb.tif")

    assert rgb_tif == root / "data" / "rgbd_upstream" / "patch_rgb.tif"
    with rasterio.open(tif) as ds:
        prof = ds.profile
        assert (prof["height"], prof["width"]) == rgb.shape[:2]
        assert prof["transform"] == rasterio.Affine(PX, 0.0, 0.0, 0.0, -PX, lf.height_m)
        assert (prof["dtype"], prof["count"], prof["crs"]) == ("float32", 1, None)
        assert np.all(ds.read(1) == GROUND_LEVEL_M)

    # The pair is what the loader streams: the map's pixels on the local frame.
    cm._ORTHO_RASTER_CACHE.clear()
    got = cm.load_ortho_or_rasters("patch", rgbd_dir=str(tif.parent), m_per_px=0.1)
    cm._ORTHO_RASTER_CACHE.clear()
    assert np.array_equal(got["rgb"], rgb)
    # The grid hangs from the frame's north edge, row 0 at y_max.
    assert got["x_min"] == 0.0
    assert got["y_min"] == pytest.approx(lf.height_m - rgb.shape[0] * PX)
    assert got["px_per_m"] == pytest.approx(1.0 / PX)


def test_a_raster_off_the_blocks_grid_is_refused_by_name(tmp_path: Path):
    lf = build_local_frame(BBOX)
    H, W = grid_shape(lf, PX)
    with rasterio.open(tmp_path / "wrong.tif", "w", driver="GTiff", width=W + 3, height=H, count=3, dtype="uint8",
                       crs=lf.crs_utm, transform=from_origin(lf.origin_e, lf.origin_n + lf.height_m, PX, PX)) as dst:
        dst.write(np.zeros((3, H, W + 3), np.uint8))
    with pytest.raises(ValueError, match="the block's frame at that cell"):
        build_block(tmp_path / "root", "patch", BBOX, tmp_path / "wrong.tif")
