import subprocess
import sys
from pathlib import Path

from benchmarks.calcity.launch import SPLIT_PROBE_ORDER, _rebuild_commands


def test_calcity_corpus_is_auto_resolvable():
    assert SPLIT_PROBE_ORDER == ("calcity",)


def test_calcity_preflight_uses_calcity_builders(tmp_path: Path):
    repo = tmp_path / "repo"
    commands = _rebuild_commands(
        map_name="calcity",
        repo=repo,
        missing_tiles=True,
        missing_rasters=True,
    )
    assert len(commands) == 2
    raster_label, raster_cmd = commands[0]
    tile_label, tile_cmd = commands[1]
    assert raster_label == "Cal City ortho rasters"
    assert "benchmarks.calcity.data_prep.build_rgbd_ortho" in raster_cmd
    assert "rasterize_citynav_native.sh" not in " ".join(raster_cmd)
    assert tile_label == "Cal City tile artifacts"
    assert tile_cmd == [
        str(repo / "benchmarks" / "calcity" / "data_prep" / "bake.sh"),
        "calcity",
    ]
    assert "rebake_citynav_local.sh" not in " ".join(tile_cmd)


def test_configured_launcher_resolves_a_released_calcity_episode():
    code = """
import json
from benchmarks.calcity.launch import _configure_shared_launcher
from benchmarks.calcity.paths import split_path
from benchmarks.citynav.adapter import parse_episode
shared = _configure_shared_launcher()
row = json.loads(split_path("calcity").read_text())[0]
episode = parse_episode(row, 0)
assert shared.resolve_split(episode.episode_id) == "calcity"
"""
    subprocess.run([sys.executable, "-c", code], check=True)
