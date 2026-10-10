"""Per-block asset loading contracts.

Every episode lives in one block: the runtime loads
``tiles/citynav/<city>_block_<N>_*`` sidecars keyed by the episode's own
map_name. These tests pin that the on-disk bake is per-block shaped (no
city-namespaced ids, block-sized bounds) and that a workdir's episode
state records that block's frame without depending on its imagery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.citynav.paths import TILES_DIR, tile_stem

BLOCK_TILE = TILES_DIR / "birmingham_block_0_graph.npz"

needs_block_bake = pytest.mark.skipif(
    not BLOCK_TILE.exists(),
    reason=f"per-block bake missing at {BLOCK_TILE}; run benchmarks/citynav/data_prep/bake_tiles.sh",
)


@needs_block_bake
def test_per_block_graph_has_no_cross_block_leakage():
    from maps.sws.graph import load_graph

    graph = load_graph(tile_stem("birmingham_block_0"))
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


# The episode-state prewrite resolves its episode against a split file, so
# these pin a block that is both baked and present in the split this repo
# iterates on.
STATE_BLOCK = "birmingham_block_1"
STATE_SPLIT = "train_seen"


def _prewrite(workdir: Path):
    """Run the episode-state prewrite for a known episode, or skip when this
    checkout carries no split data to resolve it against."""
    import json

    from benchmarks.citynav.bake.episode_state import prewrite_episode_state

    try:
        prewrite_episode_state(
            workdir=workdir,
            episode_id=f"{STATE_BLOCK}::0",
            split=STATE_SPLIT,
            write_heartbeat=False,
        )
    except (OSError, ValueError) as exc:
        pytest.skip(f"{STATE_SPLIT} split not resolvable in this checkout: {exc}")
    inputs = json.loads((workdir / "inputs.json").read_text())
    # Vacuity floor: a skip is the only way these tests may pass without
    # having resolved a real row, so prove the row was real.
    assert "#" in inputs["episode_id"], inputs["episode_id"]
    return inputs


@needs_block_bake
def test_episode_state_records_the_block_frame(tmp_path):
    """``inputs.json`` carries the episode's identity and its block's frame."""
    inputs = _prewrite(tmp_path)
    # Tile id must be the block, not the city.
    assert inputs["tile_id"] == STATE_BLOCK
    assert inputs["episode_id"].startswith(f"{STATE_BLOCK}::")
    assert inputs["split"] == STATE_SPLIT
    assert inputs["cell_size_m"] > 0
    # Bounds must match the block's own canvas (width < 600 m), not the city.
    x_min, y_min, x_max, y_max = inputs["bounds_world_xy"]
    assert (x_max - x_min) < 600
    assert (y_max - y_min) < 600
    # The dashboard serves tile-level imagery by map name, so a workdir
    # carries no copy of it and no key pointing at one.
    assert not (tmp_path / "satellite.png").exists()
    assert "satellite_image_path" not in inputs
    assert "pixel_to_world" not in inputs


@needs_block_bake
def test_episode_state_survives_a_missing_satellite_sidecar(tmp_path, monkeypatch):
    """The frame fields do not depend on the satellite bake.

    The writer used to resolve ``<map>_satellite.npz`` to embed a per-workdir
    PNG, and its call site swallowed the failure. On any host without the raw
    PLYs that took ``bounds_world_xy``, ``cell_size_m`` and ``tile_id`` down
    with an image nothing was reading — the analysis readers lost their frame
    and said nothing about it.
    """
    import importlib
    import shutil

    import benchmarks.citynav.paths as paths_mod

    # A tile tree carrying the block's meta and NOTHING else: no
    # ``_satellite.npz``, no ``_satellite.png``.
    fake_tiles = tmp_path / "tiles"
    fake_tiles.mkdir()
    shutil.copyfile(
        TILES_DIR / f"{STATE_BLOCK}_meta.json",
        fake_tiles / f"{STATE_BLOCK}_meta.json",
    )
    assert not list(fake_tiles.glob("*_satellite.*"))
    monkeypatch.setenv("SEARCH_CITYNAV_TILES", str(fake_tiles))
    # paths.TILES_DIR caches the env at import; reload to pick the override up.
    importlib.reload(paths_mod)
    try:
        inputs = _prewrite(tmp_path / "wd")
        assert inputs["tile_id"] == STATE_BLOCK
        assert inputs["cell_size_m"] > 0
        x_min, y_min, x_max, y_max = inputs["bounds_world_xy"]
        assert (x_max - x_min) > 0 and (y_max - y_min) > 0
    finally:
        # Restore the real module so later tests in this run see it.
        monkeypatch.undo()
        importlib.reload(paths_mod)
