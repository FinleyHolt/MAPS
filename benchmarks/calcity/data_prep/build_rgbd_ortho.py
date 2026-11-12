#!/usr/bin/env python3
"""Lay down the block's raster pair from its own map.

    python -m benchmarks.calcity.data_prep.build_rgbd_ortho --raster <calcity_rgb.tif> [--root ~/datasets/calcity]

The imagery the bench crops is the block's own map (the C-12's, at the
collection's native cell, laid on the block's grid and written lossless as
``<stem>_rgb.tif``). The repo ships none; the user supplies the path. For each
block this writes, under ``<root>/data/rgbd_upstream`` (where ``SEARCH_CALCITY_RGBD_DIR``
defaults):

  * ``<map>_rgb.tif`` -- the raster, copied as it is.
  * ``<map>.tif`` -- a constant height plane at ``aoi.GROUND_LEVEL_M`` (the
    flat-world contract; see ``benchmarks.calcity.data_prep.aoi``) on the
    raster's own grid, float32, whose affine carries the local-frame extent
    the ortho loader reads its scale from.

The ortho loader (``maps.perception.ortho.camera_model``) streams this pair
into its mmap cache a band of rows at a time.

Buildings are NOT stamped onto the height plane: a 2D imagery world carries
no observed geometry, and a fabricated extrusion would put parallax into
``depth = altitude - height`` that the imagery cannot corroborate.

Nothing here resamples: the raster must be the block's frame at its own
cell, ``bundle.grid_shape`` rows by columns, or it is refused by name.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from benchmarks.calcity.paths import calcity_bench
from benchmarks.calcity.data_prep.aoi import (
    AOI_BBOX,
    GROUND_LEVEL_M,
    build_local_frame,
)
from benchmarks.calcity.data_prep.bundle import grid_shape

log = logging.getLogger("build_rgbd_ortho")


def build_block(root: Path, map_name: str, bbox, raster: Path) -> tuple[Path, Path]:
    """Write one block's ``<map>.tif`` (flat height) and ``<map>_rgb.tif`` (the raster)."""
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window

    raster = Path(raster)
    with rasterio.open(raster) as src:
        px = float(abs(src.transform.a))
        W, H = src.width, src.height
        if abs(src.transform.e) != px:
            raise ValueError(
                f"{raster}: the cell is {px} x {abs(src.transform.e)} m; a block raster's is square"
            )
    lf = build_local_frame(bbox)
    want_h, want_w = grid_shape(lf, px)
    if (W, H) != (want_w, want_h):
        raise ValueError(
            f"{raster} is {W}x{H} at {px} m; the block's frame at that cell is {want_w}x{want_h}"
        )

    # Task-frame affine: row 0 = runtime y_max = geographic north.
    transform_local = Affine(px, 0.0, 0.0, 0.0, -px, lf.height_m)

    rgbd_dir = root / "data" / "rgbd_upstream"
    rgbd_dir.mkdir(parents=True, exist_ok=True)
    tif_path = rgbd_dir / f"{map_name}.tif"
    rgb_path = rgbd_dir / f"{map_name}_rgb.tif"

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="float32",
        # Coordinates are block-local task metres, not absolute UTM.
        crs=None,
        transform=transform_local,
        compress="deflate",
        bigtiff="YES",
    ) as ds:
        band = np.full((min(1024, H), W), GROUND_LEVEL_M, dtype=np.float32)
        for r0 in range(0, H, 1024):
            rows = min(1024, H - r0)
            ds.write(band[:rows], 1, window=Window(0, r0, W, rows))
    log.info("wrote %s (%dx%d, flat %.1f m)", tif_path, W, H, GROUND_LEVEL_M)

    if raster.resolve() != rgb_path.resolve():
        shutil.copyfile(raster, rgb_path)
        log.info("copied %s -> %s", raster, rgb_path)
    return tif_path, rgb_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S"
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=calcity_bench(os.environ).root)
    parser.add_argument(
        "--raster",
        type=Path,
        required=True,
        help="the block's RGB GeoTIFF at the cell the bench runs on",
    )
    args = parser.parse_args(argv)
    build_block(args.root, "calcity", AOI_BBOX, args.raster)
    return 0


if __name__ == "__main__":
    sys.exit(main())
