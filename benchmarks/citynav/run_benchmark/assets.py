"""Baked per-block assets and graph loading."""

from __future__ import annotations

from benchmarks.citynav.eval import WorldBounds
from benchmarks.citynav.paths import (
    assert_current_tile_schema,
    rgbd_png,
    rgbd_tiff,
    tile_stem,
)
from dataclasses import dataclass
from pathlib import Path
from maps.perception.observation import BlockObservation, load_block_rasters
from maps.sws.graph import load_graph
import json
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
    # Per-block ground elevation lookup for AGL-aware FPV LOD selection.
    # Subtile mode wires a constant; full mode uses a (city, bbox) → z
    # lookup that walks the per-block roster. ``None`` falls back to the
    # FPV path's default constant.
    ground_z_for_pose: object = (
        None  # Optional[Callable[[tuple[float, float]], float | None]]
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


def _load_block_assets(map_name: str, cell_size_m: float) -> BlockAssets:
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

    # The features dict is the OSM bake's sidecar, never rebuilt from
    # objects.json (that would hand the reasoner ground-truth geometry).
    feat_path = Path(f"{stem}_features.json")
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Missing {feat_path}: re-bake this block "
            f"(benchmarks/citynav/data_prep/bake_tiles.sh)."
        )
    base_features = json.loads(feat_path.read_text())
    assert_current_tile_schema(base_features, str(feat_path))
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


def load_episode_assets(map_name: str, cell_size_m: float) -> BlockAssets:
    """Load the episode's per-block bakes."""
    return _load_block_assets(map_name, cell_size_m)


def load_episode_graph(stem: str):
    """Load the ObjectGraph for the episode's block: ``load_graph(<block>)``."""
    return load_graph(stem)
