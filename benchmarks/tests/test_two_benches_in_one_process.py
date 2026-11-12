"""CityNav and Cal City stand side by side in one process: each is a value,
so the engine hydrates an episode of each from its own bake, and each run
names, stamps and scores as its own bench."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from benchmarks.calcity.paths import calcity_bench
from benchmarks.citynav.bake.city import bake_block
from benchmarks.citynav.eval import WorldBounds
from benchmarks.citynav.paths import citynav_bench
from benchmarks.citynav.run_benchmark import episode_runner
from benchmarks.run_record.config import load_config
from benchmarks.run_record.episode_workdir import mint_workdir
from benchmarks.run_record.run_manifest import run_env_stamp
from maps.settings import from_environ

BLOCK = "birmingham_block_0"


def _square(x0: float, y0: float, side: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side], [x0, y0]]
        ],
    }


@pytest.fixture
def benches(tmp_path):
    """A CityNav block bake holding a church and a Cal City bake holding a
    barn, each under its own bench's tiles directory."""
    city_tiles, cal_tiles = tmp_path / "citynav_tiles", tmp_path / "calcity_tiles"
    bake_block(
        BLOCK,
        [({"name": "Elm Church", "building": "church", "osm_id": "w1"}, _square(10, 10, 8))],
        city_tiles,
        bounds=WorldBounds(x_min=0.0, y_min=0.0, x_max=40.0, y_max=40.0),
    )
    bake_block(
        "calcity",
        [({"name": "North Barn", "building": "barn", "osm_id": "w2"}, _square(20, 20, 6))],
        cal_tiles,
        bounds=WorldBounds(x_min=0.0, y_min=0.0, x_max=60.0, y_max=60.0),
    )
    citynav = citynav_bench(
        {"SEARCH_CITYNAV_TILES": str(city_tiles), "SEARCH_CITYNAV_RGBD_DIR": str(tmp_path / "a")}
    )
    calcity = calcity_bench(
        {"SEARCH_CALCITY_TILES": str(cal_tiles), "SEARCH_CALCITY_RGBD_DIR": str(tmp_path / "b")}
    )
    return citynav, calcity


def _hydrate(bench, map_name: str, config: str, monkeypatch):
    engine = bench.engine
    # An authored start carries no height: a CityNav row states its own.
    episode = dataclasses.replace(
        engine.authored_episode(f"{map_name}::t", (5.0, 5.0)), start_z=120.0
    )
    seen = SimpleNamespace(set_concepts=lambda concepts: None)
    monkeypatch.setattr(episode_runner, "make_perception", lambda settings: seen)
    cfg = load_config(config)
    return engine.build_episode(
        cfg,
        episode,
        engine.load_assets(bench, map_name),
        bench,
        settings=from_environ(bench.environ),
    )


def test_an_episode_of_each_bench_is_hydrated_from_its_own_bake_in_one_process(
    benches, monkeypatch
):
    citynav, calcity = benches
    # CityNav first, then Cal City, then CityNav again: no order selects a bench for the process.
    first = _hydrate(citynav, BLOCK, "configs/citynav.yaml", monkeypatch)
    second = _hydrate(calcity, "calcity", "configs/calcity.yaml", monkeypatch)
    third = _hydrate(citynav, BLOCK, "configs/citynav.yaml", monkeypatch)
    for h, bench, name, shape in (
        (first, citynav, "Elm Church", (20, 20)),
        (second, calcity, "North Barn", (30, 30)),
        (third, citynav, "Elm Church", (20, 20)),
    ):
        assert h.bench is bench and h.assets.grid_shape == shape
        names = {n.name for n in h.world.state.runtime_graph.nodes.values()}
        assert names == {name}
    # A block one bench bakes is not the other's.
    with pytest.raises(FileNotFoundError, match="calcity_meta.json"):
        citynav.engine.load_assets(citynav, "calcity")
    with pytest.raises(FileNotFoundError, match=f"{BLOCK}_meta.json"):
        calcity.engine.load_assets(calcity, BLOCK)


def test_each_bench_names_scores_and_stamps_its_own_runs(benches, tmp_path):
    citynav, calcity = benches
    assert (citynav.success_criterion, calcity.success_criterion) == ("ring", "shape")
    assert (citynav.default_split, calcity.default_split) == ("val_seen", "calcity")
    assert "_citynav_" in mint_workdir(tmp_path, "e::1#1", benchmark=citynav.name).name
    assert "_calcity_" in mint_workdir(tmp_path, "e::1#1", benchmark=calcity.name).name
    # Each run records its own bake and raster: Cal City's under the shared runner's names, as a child inherits them.
    assert run_env_stamp(citynav) == {
        "SEARCH_CITYNAV_TILES": str(citynav.tiles_dir),
        "SEARCH_CITYNAV_RGBD_DIR": str(citynav.rgbd_dir),
    }
    assert run_env_stamp(calcity) == {
        "SEARCH_CITYNAV_TILES": str(calcity.tiles_dir),
        "SEARCH_CITYNAV_RGBD_DIR": str(calcity.rgbd_dir),
    }
    assert citynav.tiles_dir != calcity.tiles_dir and citynav.bake_sha256() != calcity.bake_sha256()
    assert (
        "SEARCH_BENCHMARK_NAME" not in citynav.environ
        and calcity.environ["SEARCH_BENCHMARK_NAME"] == "calcity"
    )
