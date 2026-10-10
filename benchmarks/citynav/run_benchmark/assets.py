"""Baked per-block assets and graph loading."""

from __future__ import annotations

from benchmarks.citynav.adapter import build_features
from benchmarks.citynav.eval import WorldBounds
from benchmarks.citynav.paths import (
    assert_current_tile_schema,
    rgbd_png,
    rgbd_tiff,
    tile_stem,
)
from benchmarks.citynav.run_benchmark.config import RunConfig
from dataclasses import dataclass
from pathlib import Path
from maps.perception.observation import BlockObservation, load_block_rasters
from maps.sws.graph import load_graph
import json
import os
import logging

log = logging.getLogger("run_benchmark")


@dataclass
class BlockAssets:
    """Prebaked + rasterized state shared across episodes on the same block."""

    map_name: str
    bounds: WorldBounds
    cell_size_m: float
    grid_shape: tuple[int, int]
    base_features: dict  # Prebaked features dict, loaded once per block.
    raster: BlockObservation
    satellite_view: object | None = None  # SatelliteView from _satellite.npz
    # if the bake produced one; the union pre-segment in the parse-first
    # compiler reads it once per turn. None disables segmentation for
    # this tile.
    # Per-block ground elevation lookup for AGL-aware FPV LOD selection.
    # Subtile mode wires a constant; full mode uses a (city, bbox) → z
    # lookup that walks the per-block roster. ``None`` falls back to the
    # FPV path's default constant.
    ground_z_for_pose: object = (
        None  # Optional[Callable[[tuple[float, float]], float | None]]
    )


def _assert_view_matches_meta(
    *,
    map_name: str,
    view: object,
    meta_grid_shape: tuple[int, int],
    meta_bounds: WorldBounds,
    meta_cell_size_m: float,
) -> None:
    """Hard invariant: SatelliteView's grid_shape + ground extent + cell
    size must match the tile meta.json. Stale satellite npz files (e.g.
    baked at 5 m/cell while the meta has migrated to 2 m/cell) silently
    drop SAM detections at projection by clipping cells against the
    view's stored grid_shape. We trip on that early instead of letting
    runs proceed with empty perception in the upper portion of the tile.
    """
    view_shape = getattr(view, "grid_shape", None)
    if view_shape is None or tuple(int(v) for v in view_shape) != tuple(
        int(v) for v in meta_grid_shape
    ):
        raise RuntimeError(
            f"satellite view / meta mismatch on {map_name!r}: "
            f"view.grid_shape={view_shape} vs meta.grid_shape={meta_grid_shape}. "
            f"Re-bake the _satellite.npz at the current cell_size_m: "
            f'`python -c "from benchmarks.citynav.bake.satellite import bake_block; '
            f"bake_block('{map_name}')\"`."
        )
    p2w = getattr(view, "pixel_to_world", None)
    if p2w is not None and len(p2w) >= 4:
        ground_w_m = float(p2w[2])
        ground_h_m = float(p2w[3])
        meta_w = float(meta_bounds.x_max - meta_bounds.x_min)
        meta_h = float(meta_bounds.y_max - meta_bounds.y_min)
        # Allow up to 5 m absolute tolerance — render_from_ply pads the
        # ground extent slightly. We just want to catch the 5x scale bug
        # (view = 245 vs meta = 240 is fine; view = 445 vs meta = 240 is not).
        if abs(ground_w_m - meta_w) > 5.0 or abs(ground_h_m - meta_h) > 5.0:
            raise RuntimeError(
                f"satellite view / meta ground-extent mismatch on {map_name!r}: "
                f"view ground=({ground_w_m}, {ground_h_m}) vs "
                f"meta ground=({meta_w}, {meta_h}). Re-bake the _satellite.npz."
            )
    # Sanity-check view-derived cell size against meta's cell_size_m. A
    # stale 5 m bake on a 2 m meta would have ground_h_m/grid_height ≈ 5,
    # not 2. Tolerance is loose because of the ground-extent padding.
    if p2w is not None and len(p2w) >= 4 and meta_grid_shape[0] > 0:
        view_cell_h = float(p2w[3]) / float(meta_grid_shape[0])
        if abs(view_cell_h - meta_cell_size_m) > 0.5:
            raise RuntimeError(
                f"satellite view cell size disagrees with meta on {map_name!r}: "
                f"view-derived {view_cell_h:.2f} m/cell vs meta "
                f"{meta_cell_size_m:.2f} m/cell. Re-bake the _satellite.npz."
            )


def _assert_descriptions_wall(features: object, source_path: str) -> None:
    """CityRefer descriptions-wall regression guard. Walks the loaded
    features dict (the LLM-bound catalog) and aborts loudly if any
    nested entry carries a ``description`` or ``descriptions`` field.

    The per-episode operator instruction is one item of the CityRefer
    object's NL description list. If the bake (or any upstream change)
    ever surfaces ``descriptions`` into the LLM-bound catalog, the
    ranker can string-match the instruction back to the catalog node
    that contains it and recover ground-truth position at rank 0 --
    silently flipping SR toward ~100% via label leak. The load-bearing
    wall is ``benchmarks/citynav/bake/city.py`` (descriptions
    stripped at bake time); this is a defense-in-depth assertion that
    catches a regression at load time, before any LLM call.
    """

    def _walk(obj: object, path: str) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key in ("description", "descriptions"):
                    raise RuntimeError(
                        f"CityRefer descriptions-wall REGRESSION at "
                        f"{source_path}{path}: field {key!r} reached the "
                        f"LLM-bound features catalog. This is a "
                        f"ground-truth label leak (the agent's per-episode "
                        f"instruction is one item of this list; the ranker "
                        f"would recover GT position by string-match). "
                        f"Restore the bake-time strip in "
                        f"bake/city.py; do NOT rely on the "
                        f"runtime denylist alone."
                    )
                _walk(val, f"{path}/{key}")
        elif isinstance(obj, list):
            for i, val in enumerate(obj):
                _walk(val, f"{path}[{i}]")

    _walk(features, "")


def _load_block_assets(
    map_name: str, objects_for_map: list, cell_size_m: float
) -> BlockAssets:
    stem = tile_stem(map_name)
    meta_path = Path(f"{stem}_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run benchmarks/citynav/data_prep/bake_tiles.sh for this block first."
        )
    meta = json.loads(meta_path.read_text())
    bounds = WorldBounds(
        x_min=meta["bounds"]["x_min"],
        y_min=meta["bounds"]["y_min"],
        x_max=meta["bounds"]["x_max"],
        y_max=meta["bounds"]["y_max"],
    )
    H, W = meta["grid_shape"]
    grid_shape = (int(H), int(W))

    # The tile's baked cell_size_m is authoritative — bounds + grid_shape
    # in the meta are computed against it, so any cfg-supplied value that
    # disagrees is a bug (would scale cell→world by the wrong factor and
    # send the planner off-tile). Prefer meta; fall back to the cfg arg
    # only when meta is missing the field (older bakes).
    meta_cell_size = meta.get("cell_size_m")
    if meta_cell_size is not None:
        meta_cell_size = float(meta_cell_size)
        if abs(meta_cell_size - cell_size_m) > 1e-6:
            log.warning(
                "  %s: meta cell_size_m=%.3f overrides cfg cell_size_m=%.3f",
                map_name,
                meta_cell_size,
                cell_size_m,
            )
        cell_size_m = meta_cell_size

    # Features dict comes from the baked sidecar if present; else rebuild.
    feat_path = Path(f"{stem}_features.json")
    if feat_path.exists():
        base_features = json.loads(feat_path.read_text())
        assert_current_tile_schema(base_features, str(feat_path))
    else:
        # C1 FAIL-LOUD (LEAK L1 re-verifier caveat, 2026-07-04): the rebuild
        # feeds GT objects.json geometry/names/classes into base_features,
        # which reach the reasoner. That is a bake-integrity hole, not a
        # designed path — under real-SAM GT isolation (SEARCH_SKIP_VLM absent,
        # sam), a MISSING _features.json must ABORT, never silently
        # fall back to GT. The diagnostic opt-out still permits it.
        if (os.environ.get("SEARCH_REALSTACK_GT_ISOLATION") == "1"
                and os.environ.get("SEARCH_ALLOW_GT_DIAGNOSTIC") != "1"):
            raise FileNotFoundError(
                f"Missing {feat_path} under real-SAM GT isolation: refusing to "
                f"rebuild base_features from GT objects.json (would leak GT "
                f"geometry/names to the reasoner — spec C1). Re-bake this block "
                f"(bake_city --osm-blocks) so the OSM _features.json exists, or "
                f"set SEARCH_ALLOW_GT_DIAGNOSTIC=1 for a deliberate diagnostic."
            )
        base_features, _ = build_features(
            objects_for_map, bounds, grid_shape, cell_size_m
        )
    _assert_descriptions_wall(base_features, str(feat_path))

    png_path = rgbd_png(map_name)
    tiff_path = rgbd_tiff(map_name)
    # Infer the raster's geotransform from the height TIFF when possible.
    # CityNav's rasterize.py writes at 10 px/m centered on the point cloud's
    # extent, which can be offset from (and smaller than) the block's
    # nominal bounds. Falling back to 1 px/m + bounds corner is fine for
    # the synthetic-only path (no real PNG present).
    raster_x_min = bounds.x_min
    raster_y_min = bounds.y_min
    raster_px_per_m = 1.0
    if png_path.exists():
        try:
            import rasterio  # type: ignore[import-not-found]

            with rasterio.open(str(tiff_path)) as ds:
                t = ds.transform
                raster_px_per_m = 1.0 / float(t.a)  # a = pixel x-size (m/px)
                raster_x_min = float(t.c)  # c = upper-left x
                # upper-left y − height × pixel_size_y (a.e is negative)
                raster_y_min = float(t.f + ds.height * t.e)
        except Exception as e:
            # The real upstream raster is present (png_path.exists()), so the
            # 1 px/m + bounds-corner default set above is the WRONG georef:
            # rasterize.py writes 10 px/m at an offset origin, so the default
            # mislocates every crop ~10x and the cropclient observation is
            # garbage (detections never land on targets -> every episode
            # no-commits). That default is only valid for the synthetic path
            # (no PNG, which never enters this branch). Fail loud rather than
            # silently produce broken observations.
            raise RuntimeError(
                f"{map_name}: upstream raster {png_path.name} is present but "
                f"its TIFF geotransform could not be read ({e}). The "
                f"ortho backend needs the true georef; install the "
                f"perception 'ortho' extra (rasterio>=1.3)."
            ) from e

    raster = load_block_rasters(
        map_name=map_name,
        png_path=png_path,
        tiff_path=tiff_path,
        x_min=raster_x_min,
        y_min=raster_y_min,
        px_per_m=raster_px_per_m,
        grid_shape=grid_shape,
    )
    sat_path = Path(f"{stem}_satellite.npz")
    satellite_view: object | None = None
    if sat_path.exists():
        try:
            from benchmarks.citynav.satellite_view import load_satellite_view

            satellite_view = load_satellite_view(sat_path)
        except Exception as exc:
            log.warning("  %s: failed to load satellite view: %s", map_name, exc)
        else:
            _assert_view_matches_meta(
                map_name=map_name,
                view=satellite_view,
                meta_grid_shape=grid_shape,
                meta_bounds=bounds,
                meta_cell_size_m=cell_size_m,
            )

    from maps.perception.ortho.ground_level import GROUND_LEVEL

    block_ground_z = GROUND_LEVEL.get(map_name)

    def _ground_z_subtile(_pose_xy: tuple[float, float]) -> float | None:
        return block_ground_z

    return BlockAssets(
        map_name=map_name,
        bounds=bounds,
        cell_size_m=cell_size_m,
        grid_shape=grid_shape,
        base_features=base_features,
        raster=raster,
        satellite_view=satellite_view,
        ground_z_for_pose=_ground_z_subtile,
    )


def city_root_from_map_name(map_name: str) -> str:
    """``"birmingham_block_3"`` -> ``"birmingham"``.

    Episode JSONs name the block; the city-wide bake stems are the bare
    city. A map_name that is already a city stem (no ``_block_``) returns
    unchanged.
    """
    if "_block_" in map_name:
        return map_name.split("_block_")[0]
    return map_name


def load_episode_assets(
    map_name: str,
    objects_for_map: list,
    cell_size_m: float,
) -> BlockAssets:
    """Load the episode's per-block bakes."""
    return _load_block_assets(map_name, objects_for_map, cell_size_m)


def load_episode_graph(stem: str):
    """Load the ObjectGraph for the episode's block: ``load_graph(<block>)``."""
    return load_graph(stem)
