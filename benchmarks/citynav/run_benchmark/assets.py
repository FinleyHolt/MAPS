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


def load_episode_assets(map_name: str, cell_size_m: float) -> BlockAssets:
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

    # The tile's baked cell_size_m is authoritative: bounds + grid_shape in
    # the meta are computed against it.
    cell_size_m = float(meta.get("cell_size_m", cell_size_m))

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

    return BlockAssets(
        map_name=map_name,
        bounds=bounds,
        cell_size_m=cell_size_m,
        grid_shape=grid_shape,
        base_features=base_features,
    )
