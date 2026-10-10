"""The map rendered through the camera at a pose, and north-up over a window: what a frame should show.

The registration raster is a directory holding the block's ``<map>_rgb.tif``
beside its ``<map>.tif`` on the bake's grid, the pair the bench's rasters
come in (``benchmarks.calcity.data_prep.build_rgbd_ortho``), read through the
one ortho loader the bench crops with. The twin renders the frames it serves
off the same renderer, so what the twin fabricates is what registration
matches against.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from benchmarks.flight.tile_georef import Bounds, map_bounds

#: The block's bare-earth surface beside the bake, ``<map>_dem.tif`` on the
#: bake's grid in the map's MSL; absent, the map is flat at ``FLOOR_MSL_M``.
TERRAIN_SUFFIX = "_dem.tif"


def load_raster(rgbd_dir: str | Path, map_name: str):
    """``(rgb, bounds, px_per_m)`` for a raster directory, the rgb memory-mapped; None without one."""
    from maps.perception.ortho.camera_model import load_ortho_or_rasters  # noqa: PLC0415

    bake = load_ortho_or_rasters(map_name, rgbd_dir=str(rgbd_dir))
    if bake is None:
        return None
    rgb, ppm = bake["rgb"], float(bake["px_per_m"])
    x0, y0 = float(bake["x_min"]), float(bake["y_min"])
    return rgb, Bounds(x0, y0, x0 + rgb.shape[1] / ppm, y0 + rgb.shape[0] / ppm), ppm


@dataclass(frozen=True)
class Terrain:
    """A height raster in bake-local metres, row 0 the north edge, sampled by
    nearest cell; a sample off the raster takes the nearest edge cell."""

    grid: np.ndarray
    x_min: float
    y_min: float
    cell_m: float

    def elevation(self, x, y) -> np.ndarray:
        h, w = self.grid.shape
        col = np.round((np.asarray(x, dtype=np.float64) - self.x_min) / self.cell_m).astype(np.int64)
        row = (h - 1) - np.round((np.asarray(y, dtype=np.float64) - self.y_min) / self.cell_m).astype(np.int64)
        return self.grid[np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)].astype(np.float64)


@lru_cache(maxsize=4)
def terrain(map_name: str, tiles_dir: str) -> Terrain | None:
    """The map's bare-earth surface, read once per process; None without ``<map>_dem.tif``."""
    path = Path(tiles_dir) / f"{map_name}{TERRAIN_SUFFIX}"
    if not path.is_file():
        return None
    import rasterio  # noqa: PLC0415

    with rasterio.open(path) as src:
        grid = src.read(1).astype(np.float64)
        cell = float(src.res[0])
    bounds = map_bounds(tiles_dir, map_name)
    if not np.isfinite(grid).all():
        grid = np.where(np.isfinite(grid), grid, float(np.nanmedian(grid)))
    return Terrain(grid=grid, x_min=float(bounds.x_min), y_min=float(bounds.y_min), cell_m=cell)


def ground_msl(terrain: Terrain | None, floor_msl_m: float, xy) -> float:
    """The ground under ``xy`` in the map's MSL: the map's ``terrain`` where it carries one, else its floor."""
    if terrain is None:
        return float(floor_msl_m)
    return float(terrain.elevation(np.asarray([xy[0]], dtype=np.float64), np.asarray([xy[1]], dtype=np.float64))[0])


def window_render(window: tuple[float, float, float, float], shape: tuple[int, int], ortho) -> np.ndarray | None:
    """The map north-up over ``window`` (``(x0, y0, x1, y1)`` bake-local metres)
    at ``shape`` (rows, cols), row 0 the window's north edge; None where the
    window reaches off the raster."""
    import cv2  # noqa: PLC0415

    rgb, bounds, ppm = ortho
    x0, y0, x1, y1 = window
    rows, cols = rgb.shape[:2]
    c0, c1 = int(round((x0 - bounds.x_min) * ppm)), int(round((x1 - bounds.x_min) * ppm))
    r0, r1 = int(round((rows - 1) - (y1 - bounds.y_min) * ppm)), int(round((rows - 1) - (y0 - bounds.y_min) * ppm))
    if c0 < 0 or r0 < 0 or c1 > cols or r1 > rows or c1 <= c0 or r1 <= r0:
        return None
    return cv2.resize(np.ascontiguousarray(rgb[r0:r1, c0:c1, :3]), (int(shape[1]), int(shape[0])),
                      interpolation=cv2.INTER_AREA)


def renderer(ortho):
    """The expected view off ``ortho``: ``(sensor, pose_xy, agl_m, yaw_deg,
    roll_deg, pitch_deg) -> (H, W, 3)`` uint8 over flat ground, ``pitch_deg``
    the projector's (nose down). Raises ``ValueError`` when the footprint
    leaves the raster or a corner reaches the horizon."""
    from maps.perception.ortho.sensor_frame import render_frame  # noqa: PLC0415

    rgb, bounds, ppm = ortho

    def render(sensor, pose_xy, agl_m: float, yaw_deg: float, roll_deg: float = 0.0,
               pitch_deg: float = 0.0) -> np.ndarray:
        frame, _stats = render_frame(rgb, bounds, ppm, sensor=sensor, pose_xy=pose_xy, agl_m=agl_m,
                                     yaw_deg=yaw_deg, airframe_roll_deg=roll_deg, airframe_pitch_deg=pitch_deg)
        return np.ascontiguousarray(np.asarray(frame)[..., :3])

    return render
