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
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=W,
        height=H,
        count=3,
        dtype="uint8",
        crs=lf.crs_utm,
        transform=from_origin(lf.origin_e, lf.origin_n + lf.height_m, px, px),
        compress="deflate",
    ) as dst:
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
    got = cm.load_ortho_or_rasters("patch", rgbd_dir=str(tif.parent))
    cm._ORTHO_RASTER_CACHE.clear()
    assert np.array_equal(got["rgb"], rgb)
    # The grid hangs from the frame's north edge, row 0 at y_max.
    assert got["x_min"] == 0.0
    assert got["y_min"] == pytest.approx(lf.height_m - rgb.shape[0] * PX)
    assert got["px_per_m"] == pytest.approx(1.0 / PX)


def test_a_raster_off_the_blocks_grid_is_refused_by_name(tmp_path: Path):
    lf = build_local_frame(BBOX)
    H, W = grid_shape(lf, PX)
    with rasterio.open(
        tmp_path / "wrong.tif",
        "w",
        driver="GTiff",
        width=W + 3,
        height=H,
        count=3,
        dtype="uint8",
        crs=lf.crs_utm,
        transform=from_origin(lf.origin_e, lf.origin_n + lf.height_m, PX, PX),
    ) as dst:
        dst.write(np.zeros((3, H, W + 3), np.uint8))
    with pytest.raises(ValueError, match="the block's frame at that cell"):
        build_block(tmp_path / "root", "patch", BBOX, tmp_path / "wrong.tif")


def _write(path: Path, crs, transform, H: int, W: int) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=W,
        height=H,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(np.zeros((3, H, W), np.uint8))
    return path


def test_a_raster_in_another_crs_is_refused_naming_both(tmp_path: Path):
    """A right-sized raster in degrees would shift every crop off its target."""
    lf = build_local_frame(BBOX)
    H, W = grid_shape(lf, PX)
    path = _write(tmp_path / "wgs84.tif", "EPSG:4326", from_origin(-117.97, 35.12, PX, PX), H, W)
    with pytest.raises(ValueError, match=r"EPSG:4326.*EPSG:32611"):
        build_block(tmp_path / "root", "patch", BBOX, path)
    assert not (tmp_path / "root").exists()


def test_a_raster_shifted_off_the_frame_is_refused_naming_both_bounds(tmp_path: Path):
    lf = build_local_frame(BBOX)
    H, W = grid_shape(lf, PX)
    west = from_origin(lf.origin_e - 100_000, lf.origin_n + lf.height_m, PX, PX)
    path = _write(tmp_path / "west.tif", lf.crs_utm, west, H, W)
    with pytest.raises(ValueError, match=rf"{lf.origin_e - 100_000:,.2f}.*{lf.origin_e:,.2f}"):
        build_block(tmp_path / "root", "patch", BBOX, path)
    assert not (tmp_path / "root").exists()


def test_a_raster_within_a_cell_of_the_frame_is_accepted(tmp_path: Path):
    """gdalwarp snaps the extent to whole cells; that sub-cell offset is the frame."""
    lf = build_local_frame(BBOX)
    H, W = grid_shape(lf, PX)
    nudged = from_origin(lf.origin_e + 0.4 * PX, lf.origin_n + lf.height_m - 0.4 * PX, PX, PX)
    path = _write(tmp_path / "nudged.tif", "EPSG:32611", nudged, H, W)
    build_block(tmp_path / "root", "patch", BBOX, path)


def _fatal_one_line(capsys, *argv: str) -> str:
    from benchmarks.calcity.data_prep import build_rgbd_ortho

    assert build_rgbd_ortho.main(list(argv)) == 1
    err = capsys.readouterr().err.strip()
    assert err.startswith("FATAL: ") and "\n" not in err and "Traceback" not in err
    return err


def test_a_missing_raster_exits_1_with_one_fatal_line(tmp_path: Path):
    """Run as the operator runs it: GDAL's own log of the failed open stays off stderr too."""
    import subprocess
    import sys

    argv = ["--root", str(tmp_path), "--raster", str(tmp_path / "absent.tif")]
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.calcity.data_prep.build_rgbd_ortho", *argv],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    err = proc.stderr.strip()
    assert proc.returncode == 1
    assert err.startswith("FATAL: ") and "\n" not in err and "Traceback" not in err
    assert "absent.tif" in err


def test_a_wrong_size_raster_exits_1_with_one_fatal_line(tmp_path: Path, capsys):
    lf = build_local_frame(BBOX)
    path = _write(
        tmp_path / "small.tif",
        lf.crs_utm,
        from_origin(lf.origin_e, lf.origin_n + lf.height_m, PX, PX),
        8,
        8,
    )
    err = _fatal_one_line(capsys, "--root", str(tmp_path), "--raster", str(path))
    assert "small.tif" in err
