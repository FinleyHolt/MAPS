import subprocess
import sys
from pathlib import Path

from benchmarks.calcity.launch import SPLIT_PROBE_ORDER, _rebuild_commands


def test_calcity_corpus_is_auto_resolvable():
    assert SPLIT_PROBE_ORDER == ("calcity",)


def test_calcity_preflight_bakes_tiles_with_the_calcity_baker():
    commands = _rebuild_commands(map_name="calcity", missing_tiles=True)
    assert commands == [
        (
            "Cal City tile artifacts",
            [sys.executable, "-m", "benchmarks.calcity.data_prep.bake_tiles",
             "--blocks", "calcity"],
        )
    ]
    assert _rebuild_commands(map_name="calcity", missing_tiles=False) == []


def test_a_missing_raster_pair_fails_naming_the_raster_and_its_builder(
    tmp_path: Path, monkeypatch, capsys
):
    """The block's raster is the user's to supply, so preflight never
    builds it: it names ``<map>_rgb.tif`` and the command that lays the
    pair down from the user's own raster."""
    import pytest

    from benchmarks.calcity import launch

    monkeypatch.setattr(launch, "CALCITY_NATIVE_RGBD_DIR", tmp_path)
    monkeypatch.setattr(launch, "tile_stem", lambda m: str(tmp_path / m))
    for suffix in ("_meta.json", "_features.json", "_graph.npz", "_nodes.json"):
        (tmp_path / f"calcity{suffix}").write_text("{}")
    (tmp_path / "calcity.tif").write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        launch.preflight_artifacts(map_name="calcity", config_path=Path("c.yaml"))
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert str(tmp_path / "calcity_rgb.tif") in err
    assert "benchmarks.calcity.data_prep.build_rgbd_ortho --raster" in err
    assert str(tmp_path / "calcity.tif") + "\n" not in err


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
