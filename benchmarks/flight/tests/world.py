"""A synthetic Cal City for the flight tests (a georeferenced bake frame and a raster of buildings on it),
and the stand-ins they share for a vehicle, a hydrated mission and its session."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

#: The world: 1.2 km square at half a metre a pixel.
SIZE_M = 1200.0
PX_M = 0.5
#: Where its origin sits in UTM 11N, inside Cal City's AOI.
ORIGIN_E, ORIGIN_N = 410000.0, 3886000.0
#: The C-12's camera at a quarter of its pixels, the fields of view held.
SENSOR = {"base": "oryx_orx32_25mm_nadir", "image_width": 1024, "image_height": 768}
AGL_M = 914.4


def build(root: Path) -> tuple[Path, Path]:
    """``(tiles_dir, raster_dir)`` under ``root``: ``calcity_meta.json`` with
    bounds and georef, an empty ``calcity_features.json``, and the
    ``calcity_rgb.tif`` / ``calcity.tif`` pair."""
    import cv2

    tiles, raster = root / "tiles", root / "raster"
    tiles.mkdir(parents=True)
    raster.mkdir(parents=True)
    (tiles / "calcity_meta.json").write_text(
        json.dumps(
            {
                "map_name": "calcity",
                "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": SIZE_M, "y_max": SIZE_M},
                "georef": {"crs_utm_epsg": 32611, "origin_e": ORIGIN_E, "origin_n": ORIGIN_N},
            }
        )
    )
    (tiles / "calcity_features.json").write_text(json.dumps({"prior": []}))
    n = int(SIZE_M / PX_M)
    rng = np.random.default_rng(7)
    rgb = np.full((n, n, 3), 140, np.uint8)
    rgb += rng.integers(0, 12, (n, n, 1), dtype=np.uint8)
    for _ in range(1800):
        x, y = rng.integers(0, n - 40, 2)
        w, h = rng.integers(10, 40, 2)
        cv2.rectangle(
            rgb,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            tuple(int(c) for c in rng.integers(20, 255, 3)),
            -1,
        )
    for k in range(0, n, 160):
        cv2.line(rgb, (0, k), (n, k + 30), (60, 60, 60), 6)
        cv2.line(rgb, (k, 0), (k + 20, n), (60, 60, 60), 6)
    write_raster_pair(raster, rgb, PX_M)
    return tiles, raster


def texture(n: int, rects: int) -> np.ndarray:
    """An ``n`` px square of grey ground under ``rects`` small coloured roofs."""
    import cv2

    rng = np.random.default_rng(3)
    img = np.full((n, n, 3), 150, np.uint8)
    for _ in range(rects):
        x, y = rng.integers(0, n - 30, 2)
        cv2.rectangle(
            img,
            (int(x), int(y)),
            (int(x + 12), int(y + 9)),
            tuple(int(c) for c in rng.integers(20, 255, 3)),
            -1,
        )
    return img


def write_raster_pair(raster_dir: Path, rgb: np.ndarray, px_m: float) -> None:
    """``calcity_rgb.tif`` from ``rgb`` and a flat ``calcity.tif`` height beside it,
    north-up with the south-west corner at the frame origin."""
    import rasterio
    from rasterio.transform import from_origin

    n = rgb.shape[0]
    transform = from_origin(0.0, n * px_m, px_m, px_m)
    with rasterio.open(
        raster_dir / "calcity_rgb.tif",
        "w",
        driver="GTiff",
        width=n,
        height=n,
        count=3,
        dtype="uint8",
        transform=transform,
    ) as dst:
        for b in range(3):
            dst.write(rgb[..., b], b + 1)
    with rasterio.open(
        raster_dir / "calcity.tif",
        "w",
        driver="GTiff",
        width=n,
        height=n,
        count=1,
        dtype="float32",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((1, n, n), np.float32))


def recording_legs(vehicle):
    """``vehicle`` with every leg posted to it kept in ``legs``, for a test to read."""
    vehicle.legs, post = [], vehicle.post
    vehicle.post = lambda leg: (vehicle.legs.append(leg), post(leg))[-1]
    return vehicle


def hydrated(sensor, cells=None):
    """The parts of a hydrated mission the leg reads."""
    return SimpleNamespace(
        sensor=sensor,
        state=SimpleNamespace(pose_xy=(600.0, 100.0), altitude_m=AGL_M),
        ground_z_m=0.0,
        world=SimpleNamespace(node_cells_xy=lambda token: cells),
    )


def session(h, fly):
    """The parts of a mission session a leg reads: the hydrated stack, the
    target resolution, and the flight the leg hands its observation to."""
    from benchmarks.citynav.run_benchmark.mission_engine import resolve_flight_target

    return SimpleNamespace(
        h=h, fly=fly, resolve=lambda token: resolve_flight_target(token, node_xy=lambda fid: None)
    )
