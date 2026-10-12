"""Baked per-block assets and graph loading."""

from __future__ import annotations

from benchmarks.citynav.eval import WorldBounds
from benchmarks.citynav.paths import (
    assert_current_tile_schema,
    tile_stem,
)
from dataclasses import dataclass
from pathlib import Path
import json
import logging

log = logging.getLogger("run_benchmark")


@dataclass
class BlockAssets:
    """Prebaked state shared across episodes on the same block."""

    map_name: str
    bounds: WorldBounds
    cell_size_m: float
    grid_shape: tuple[int, int]
    base_features: dict  # Prebaked features dict, loaded once per block.


def load_episode_assets(map_name: str) -> BlockAssets:
    """Load the episode's per-block bakes."""
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

    # The features dict is the OSM bake's sidecar, never rebuilt from
    # objects.json (that would hand the reasoner ground-truth geometry); the
    # bake drops the CityRefer descriptions (bake/city.py _PRIVILEGED_KEYS).
    feat_path = Path(f"{stem}_features.json")
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Missing {feat_path}: re-bake this block "
            f"(benchmarks/citynav/data_prep/bake_tiles.sh)."
        )
    base_features = json.loads(feat_path.read_text())
    assert_current_tile_schema(base_features, str(feat_path))

    return BlockAssets(
        map_name=map_name,
        bounds=bounds,
        # bounds and grid_shape are computed against the bake's own cell size.
        cell_size_m=float(meta["cell_size_m"]),
        grid_shape=grid_shape,
        base_features=base_features,
    )
