"""Baked per-block assets and graph loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmarks.bench import BAKE_OUTPUT_SUFFIXES, Bench, missing_bake
from benchmarks.citynav.eval import WorldBounds


@dataclass
class BlockAssets:
    """Prebaked state shared across episodes on the same block."""

    map_name: str
    bounds: WorldBounds
    cell_size_m: float
    grid_shape: tuple[int, int]
    base_features: dict  # Prebaked features dict, loaded once per block.


def assert_current_tile_schema(features: dict, source: str) -> None:
    """Refuse a loaded ``<block>_features.json`` that is not the current bake's
    one ``prior`` list of row dicts: a run against another bake still
    completes, with a wrong SR."""
    rows = features.get("prior", []) if isinstance(features, dict) else None
    if (
        not isinstance(rows, list)
        or set(features) - {"prior"}
        or not all(isinstance(r, dict) for r in rows)
    ):
        raise RuntimeError(
            f"{source} is not the current tile bake (one 'prior' list of tag "
            "rows); grounding against it silently produces a wrong SR. Point "
            "SEARCH_CITYNAV_TILES at the current bake (tiles/citynav_2m_osmv3) "
            "or re-bake with `benchmarks/citynav/data_prep/bake_tiles.sh`."
        )


def load_episode_assets(bench: Bench, map_name: str) -> BlockAssets:
    """Load the episode's per-block bakes from ``bench``'s tile bake."""
    stem = bench.tile_stem(map_name)
    if why := missing_bake(stem):
        raise FileNotFoundError(why)
    meta_path, feat_path = (Path(f"{stem}{s}") for s in BAKE_OUTPUT_SUFFIXES)
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
