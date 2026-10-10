"""Top-down satellite-style RGB for a CityNav block.

``render_from_ply`` samples the block's baked 0.10 m/px hole-free ortho
(``benchmarks.citynav.bake.ortho``) into a nadir view whose
``pixel_to_world`` header inverts pixels to world meters
(``world_xy_to_pixel``). The bake driver writes one ``_satellite.npz``
sidecar per block via :func:`save_satellite_view`; the runtime loader
reads it via :func:`load_satellite_view` so the per-turn union
pre-segment runs SAM 3.1 against the same RGB the bake produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.citynav.eval import WorldBounds


@dataclass
class SatelliteView:
    rgb: np.ndarray
    grid_shape: tuple[int, int]
    source: str
    upscale: int | None = None
    bounds: WorldBounds | None = None
    pixel_to_world: np.ndarray | None = None
    # ``alpha`` (H, W) uint8: 255 where the renderer stamped real pixels,
    # 0 where the canvas has no coverage (the block's PLY footprint is
    # tighter than its nominal bounds). The dashboard PNG writer renders
    # zero-alpha gaps as transparent.
    alpha: np.ndarray | None = None

    @property
    def pixel_shape(self) -> tuple[int, int]:
        h, w = self.rgb.shape[:2]
        return int(h), int(w)

    def world_xy_to_pixel(
        self, world_x: np.ndarray, world_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map world (x, y) arrays to satellite pixel (row, col).

        Bound form of the module-level :func:`world_xy_to_pixel` so a
        consumer that already holds an opaque ``SatelliteView`` (e.g.
        the robot-agnostic dashboard briefer) can project without
        importing this benchmark module."""
        return world_xy_to_pixel(self, world_x, world_y)


def render_from_ply(
    ply_path: Path,
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    *,
    altitude_m: float = 800.0,
    pixels_per_m: float = 4.0,
    point_size: float = 2.5,
    margin_m: float = 0.0,
    square: bool = False,
) -> SatelliteView:
    """True-overhead nadir view of a tile, sampled from the baked ortho.

    Looks up ``benchmarks.citynav.bake.ortho``'s 0.10 m/px hole-free mosaic for
    the block, crops the tile bounding box, and resamples to
    ``pixels_per_m``. SAM 3 sees a clean orthomosaic instead of a
    point-splatted PLY render. ``altitude_m`` and ``point_size`` are kept
    in the signature for callsite compatibility but no longer participate
    in rasterization.

    ``margin_m`` and ``square`` default to a tight rectangular crop that
    matches the block's ``_meta.json`` bounds exactly — the canonical
    per-block satellite has no padding so the dashboard overlay aligns
    cell-for-cell. Set ``margin_m`` or ``square=True`` for hero-demo
    renders that want context around the block.
    """
    del altitude_m, point_size  # ortho lookup makes these moot

    from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR
    from maps.perception.ortho.camera_model import load_ortho_or_rasters

    # The ONE ortho loader — the same one the observation, the look tool and
    # the fieldsite paths read through. Using the PLY-only loader here meant a
    # box that had the raster-baked ortho (which the runtime is perfectly
    # happy with) still could not bake a satellite, and worse, a box with both
    # could render the satellite off a different mosaic than the frames.
    ortho = load_ortho_or_rasters(
        ply_path.stem, ply_path=ply_path, rgbd_dir=CITYNAV_NATIVE_RGBD_DIR
    )
    if ortho is None:
        raise FileNotFoundError(
            f"No 0.10 m/px ortho for {ply_path.stem} (looked for a PLY bake "
            f"beside {ply_path} and rasters under {CITYNAV_NATIVE_RGBD_DIR}). "
            f"Run python -m benchmarks.citynav.bake.ortho to bake it."
        )

    H_o = ortho["rgb"].shape[0]
    W_o = ortho["rgb"].shape[1]
    px_per_m = float(ortho["px_per_m"])
    ox_min = float(ortho["x_min"])
    oy_min = float(ortho["y_min"])

    ground_w_m = bounds.width_m + 2.0 * margin_m
    ground_h_m = bounds.height_m + 2.0 * margin_m
    if square:
        ground_w_m = ground_h_m = float(max(ground_w_m, ground_h_m))
    center_x = 0.5 * (bounds.x_min + bounds.x_max)
    center_y = 0.5 * (bounds.y_min + bounds.y_max)
    half_w = ground_w_m / 2.0
    half_h = ground_h_m / 2.0

    # Output canvas spans the requested ground rectangle. The ortho's
    # world coverage is generally tighter than the block's nominal
    # ``_meta.json`` bounds (e.g. birmingham_block_0's PLY covers
    # ~144x345 m of the 245x445 m nominal canvas), so we stamp the
    # in-coverage ortho patch into the canvas at its true world
    # location and leave the rest alpha=0. That keeps the dashboard
    # overlay aligned to ``_meta`` bounds without smearing extrapolated
    # edge pixels across the unscanned region.
    image_w = int(round(ground_w_m * pixels_per_m))
    image_h = int(round(ground_h_m * pixels_per_m))
    rgb = np.zeros((image_h, image_w, 3), dtype=np.uint8)
    alpha = np.zeros((image_h, image_w), dtype=np.uint8)

    # Intersect requested world rect with ortho world rect.
    ox_max = ox_min + W_o / px_per_m
    oy_max = oy_min + H_o / px_per_m
    req_x_min = center_x - half_w
    req_x_max = center_x + half_w
    req_y_min = center_y - half_h
    req_y_max = center_y + half_h
    isect_x_min = max(req_x_min, ox_min)
    isect_x_max = min(req_x_max, ox_max)
    isect_y_min = max(req_y_min, oy_min)
    isect_y_max = min(req_y_max, oy_max)
    if isect_x_max > isect_x_min and isect_y_max > isect_y_min:
        # Ortho pixel-bbox for the intersection. Row 0 = top = max y.
        col_min = int(np.floor((isect_x_min - ox_min) * px_per_m))
        col_max = int(np.ceil((isect_x_max - ox_min) * px_per_m))
        row_max = int(np.ceil((H_o - 1) - (isect_y_min - oy_min) * px_per_m))
        row_min = int(np.floor((H_o - 1) - (isect_y_max - oy_min) * px_per_m))
        col_min = max(0, col_min)
        col_max = min(W_o, col_max)
        row_min = max(0, row_min)
        row_max = min(H_o, row_max)
        crop_rgb = ortho["rgb"][row_min:row_max, col_min:col_max]
        crop_alpha = ortho["alpha"][row_min:row_max, col_min:col_max]
        # Output pixel-bbox for the intersection.
        out_col_min = int(round((isect_x_min - req_x_min) * pixels_per_m))
        out_col_max = int(round((isect_x_max - req_x_min) * pixels_per_m))
        out_row_max = int(round((req_y_max - isect_y_min) * pixels_per_m))
        out_row_min = int(round((req_y_max - isect_y_max) * pixels_per_m))
        target_w = max(0, out_col_max - out_col_min)
        target_h = max(0, out_row_max - out_row_min)
        if target_w > 0 and target_h > 0 and crop_rgb.size > 0:
            from PIL import Image as _Image

            rgb_resized = np.asarray(
                _Image.fromarray(crop_rgb).resize(
                    (target_w, target_h), _Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )
            alpha_resized = np.asarray(
                _Image.fromarray(crop_alpha).resize(
                    (target_w, target_h), _Image.Resampling.NEAREST
                ),
                dtype=np.uint8,
            )
            rgb[
                out_row_min : out_row_min + target_h,
                out_col_min : out_col_min + target_w,
            ] = rgb_resized
            alpha[
                out_row_min : out_row_min + target_h,
                out_col_min : out_col_min + target_w,
            ] = alpha_resized

    pixel_to_world = np.array(
        [center_x, center_y, ground_w_m, ground_h_m, image_h, image_w],
        dtype=np.float64,
    )
    return SatelliteView(
        rgb=rgb,
        grid_shape=grid_shape,
        source="ply",
        bounds=bounds,
        pixel_to_world=pixel_to_world,
        alpha=alpha,
    )


def world_xy_to_pixel(
    view: SatelliteView, world_x: np.ndarray, world_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map world (x, y) arrays to satellite pixel (row, col).

    Vectorized over numpy arrays. Use this to project depth-aware world
    coordinates back onto the satellite image so per-frame SAM masks can
    be rendered at pixel-fidelity on the same image the dashboard
    displays.

    Returns float arrays of pixel rows and pixel cols (NOT clamped or
    rounded; callers cast when they need integer pixel indices).
    """
    if view.source != "ply" or view.pixel_to_world is None:
        raise ValueError("world_xy_to_pixel only defined for PLY-rendered views")
    center_x, center_y, gw, gh, h_px, w_px = view.pixel_to_world.tolist()
    if gw <= 0 or gh <= 0 or w_px <= 0 or h_px <= 0:
        raise ValueError(
            f"world_xy_to_pixel needs positive ground/image extents; got "
            f"gw={gw} gh={gh} w_px={w_px} h_px={h_px}"
        )
    xs = np.asarray(world_x, dtype=np.float64)
    ys = np.asarray(world_y, dtype=np.float64)
    pixel_col = (xs - center_x) * (w_px / gw) + 0.5 * w_px
    pixel_row = (center_y - ys) * (h_px / gh) + 0.5 * h_px
    return pixel_row, pixel_col


def save_satellite_view(path: Path, view: SatelliteView) -> None:
    """Write a SatelliteView to ``_satellite.npz`` for runtime reload.

    Persists the four fields the union pre-segment needs:
    ``rgb`` (uint8 HxWx3), ``source`` (``"ply"`` or ``"palette"``),
    ``grid_shape`` (the belief grid the tile bake settled on), and one of
    ``pixel_to_world`` (PLY) or ``upscale`` (palette) so the per-tier
    grid-projection helpers know which inverse to apply. ``bounds`` is
    saved when present so the PLY path can reconstruct world XY at load.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: dict[str, np.ndarray] = {
        "rgb": np.asarray(view.rgb, dtype=np.uint8),
        "source": np.array(view.source, dtype=object),
        "grid_shape": np.array(view.grid_shape, dtype=np.int32),
    }
    if view.upscale is not None:
        fields["upscale"] = np.array(int(view.upscale), dtype=np.int32)
    if view.pixel_to_world is not None:
        fields["pixel_to_world"] = np.asarray(view.pixel_to_world, dtype=np.float64)
    if view.alpha is not None:
        fields["alpha"] = np.asarray(view.alpha, dtype=np.uint8)
    if view.bounds is not None:
        fields["bounds"] = np.array(
            [
                view.bounds.x_min,
                view.bounds.y_min,
                view.bounds.x_max,
                view.bounds.y_max,
            ],
            dtype=np.float64,
        )
    np.savez_compressed(path, **fields)


def load_satellite_view(path: Path) -> SatelliteView:
    """Inverse of :func:`save_satellite_view`."""
    data = np.load(path, allow_pickle=True)
    rgb = np.asarray(data["rgb"], dtype=np.uint8)
    source = str(
        data["source"].item() if data["source"].shape == () else data["source"]
    )
    H, W = (int(v) for v in data["grid_shape"])
    upscale = int(data["upscale"].item()) if "upscale" in data.files else None
    pixel_to_world = (
        np.asarray(data["pixel_to_world"], dtype=np.float64)
        if "pixel_to_world" in data.files
        else None
    )
    bounds = None
    if "bounds" in data.files:
        b = data["bounds"]
        bounds = WorldBounds(
            x_min=float(b[0]),
            y_min=float(b[1]),
            x_max=float(b[2]),
            y_max=float(b[3]),
        )
    alpha = np.asarray(data["alpha"], dtype=np.uint8) if "alpha" in data.files else None
    return SatelliteView(
        rgb=rgb,
        grid_shape=(H, W),
        source=source,
        upscale=upscale,
        bounds=bounds,
        pixel_to_world=pixel_to_world,
        alpha=alpha,
    )


__all__ = [
    "SatelliteView",
    "render_from_ply",
    "world_xy_to_pixel",
    "save_satellite_view",
    "load_satellite_view",
]
