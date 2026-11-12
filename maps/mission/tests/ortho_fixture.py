"""One synthetic block raster pair on disk for the ortho tests, written once per module."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_rasters(
    root: Path,
    map_name: str,
    rgb: np.ndarray,
    height: np.ndarray,
    *,
    m_per_px: float = 0.1,
    x_min: float = 0.0,
    y_min: float = 0.0,
) -> Path:
    """``<map>.png`` + ``<map>.tif`` under ``root``, the pair the ortho loader
    builds from (alpha is any nonzero RGB channel). Returns ``root``."""
    import rasterio
    from PIL import Image
    from rasterio.transform import from_origin

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(root / f"{map_name}.png")
    h, w = height.shape
    with rasterio.open(
        root / f"{map_name}.tif",
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=1,
        dtype="float32",
        transform=from_origin(x_min, y_min + h * m_per_px, m_per_px, m_per_px),
    ) as dst:
        dst.write(height.astype(np.float32), 1)
    return root


def write_ortho(root: Path, map_name: str, *, n: int, ground_z: float, ramp: bool = True) -> Path:
    """An ``n`` x ``n`` block at 0.1 m/px: ``ramp`` pixels (row and column ramps)
    so an off-by-one window shift changes bytes, else flat grey. Full coverage;
    the loader bakes its memory map beside it on first read."""
    if ramp:
        rgb = np.zeros((n, n, 3), dtype=np.uint8)
        rgb[..., 0] = (np.arange(n, dtype=np.uint32)[:, None] % 256).astype(np.uint8)
        rgb[..., 1] = (np.arange(n, dtype=np.uint32)[None, :] % 256).astype(np.uint8)
        rgb[..., 2] = 96
    else:
        rgb = np.full((n, n, 3), 120, dtype=np.uint8)
    return write_rasters(root, map_name, rgb, np.full((n, n), ground_z + 5.0, dtype=np.float32))
