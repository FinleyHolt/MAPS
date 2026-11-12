"""Per-block asset loading contracts.

Every episode lives in one block: the runtime loads
``tiles/citynav/<city>_block_<N>_*`` sidecars keyed by the episode's own
map_name. These tests pin that the on-disk bake is per-block shaped (no
city-namespaced ids, block-sized bounds) and that a workdir's episode
state records that block's frame without depending on its imagery.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmarks.citynav.paths import citynav_bench

# The bench as this process's environment places it.
BENCH = citynav_bench(os.environ)
tile_stem = BENCH.tile_stem
BLOCK_TILE = BENCH.tiles_dir / "birmingham_block_0_features.json"

needs_block_bake = pytest.mark.skipif(
    not BLOCK_TILE.exists(),
    reason=f"per-block bake missing at {BLOCK_TILE}; run benchmarks/citynav/data_prep/bake_tiles.sh",
)


@needs_block_bake
def test_per_block_graph_has_no_cross_block_leakage():
    import json

    from maps.world_state.prior_graph import build_object_graph

    stem = tile_stem("birmingham_block_0")
    meta = json.loads(Path(f"{stem}_meta.json").read_text())
    graph = build_object_graph(
        json.loads(Path(f"{stem}_features.json").read_text()),
        cell_size_m=float(meta["cell_size_m"]),
    )
    ids = list(graph.nodes.keys())
    # The tags-only bake emits generic prior ids (road_0, building_3) with
    # CityRefer provenance in tags — never city/block-namespaced ids.
    namespaced = [i for i in ids if "birmingham_block_" in i]
    assert namespaced == [], f"per-block bake leaked namespaced ids: {namespaced[:5]}"
    # Reasonable node count for one block (the smoke bake carries a
    # handful of priors; a full bake a few hundred).
    assert 0 < len(ids) < 2000


def test_tile_stem_keeps_the_block_suffix():
    assert tile_stem("birmingham_block_0").endswith("birmingham_block_0")


@needs_block_bake
def test_per_block_meta_bounds_match_block_canvas():
    """The per-block ``_meta.json`` must describe the block's own canvas."""
    import json

    meta = json.loads(Path(f"{tile_stem('birmingham_block_0')}_meta.json").read_text())
    bounds = meta["bounds"]
    # birmingham_block_0 is a ~300m x 300m subtile, not the city's
    # 1300m x 1300m canvas.
    width = bounds["x_max"] - bounds["x_min"]
    height = bounds["y_max"] - bounds["y_min"]
    assert width < 600
    assert height < 600


def test_a_missing_bake_names_its_maps_recipe(tmp_path, monkeypatch):
    """A Cal City bake lacking a file names the Cal City recipe; a CityNav
    block names bake_tiles.sh."""
    from benchmarks.citynav.run_benchmark import assets

    (tmp_path / "calcity_meta.json").write_text("{}")
    bench = citynav_bench({"SEARCH_CITYNAV_TILES": str(tmp_path)})
    with pytest.raises(
        FileNotFoundError,
        match=r"calcity_features\.json.*benchmarks\.calcity\.data_prep\.bake_tiles",
    ):
        assets.load_episode_assets(bench, "calcity")
    with pytest.raises(
        FileNotFoundError, match=r"blk_meta\.json.*benchmarks/citynav/data_prep/bake_tiles\.sh"
    ):
        assets.load_episode_assets(bench, "blk")
