from pathlib import Path

from benchmarks.calcity import paths
from benchmarks.calcity.paths import calcity_bench
from benchmarks.citynav.paths import citynav_bench


def test_split_contract_is_calcity_owned():
    bench = calcity_bench({})
    assert bench.split_files == {"calcity": "calcity.json"}
    assert bench.split_path("calcity") == bench.splits_dir / "calcity.json"


def test_default_release_layout_is_inside_calcity_package():
    """The default falls inside the package when nothing overrides it: the
    bench of an empty environment, whatever this session's own names."""
    package = Path(paths.__file__).resolve().parent
    bench = calcity_bench({})
    assert bench.splits_dir == package / "data"
    assert bench.tiles_dir == package / "tiles"


def test_an_override_places_each_root(tmp_path):
    bench = calcity_bench(
        {
            "SEARCH_CALCITY_DATA": str(tmp_path / "d"),
            "SEARCH_CALCITY_SPLITS": str(tmp_path / "s"),
            "SEARCH_CALCITY_TILES": str(tmp_path / "t"),
            "SEARCH_CALCITY_RGBD_DIR": str(tmp_path / "r"),
        }
    )
    assert (bench.root, bench.splits_dir, bench.tiles_dir, bench.rgbd_dir) == (
        tmp_path / "d",
        tmp_path / "s",
        tmp_path / "t",
        tmp_path / "r",
    )
    assert bench.objects_json == tmp_path / "d" / "data" / "cityrefer" / "objects.json"
    # CALCITY_ROOT is the data root's fallback, and the rasters default under the root.
    fallback = calcity_bench({"CALCITY_ROOT": str(tmp_path / "f")})
    assert (
        fallback.root == tmp_path / "f"
        and fallback.rgbd_dir == tmp_path / "f" / "data" / "rgbd_upstream"
    )


def test_citynav_registry_contains_no_calcity_splits():
    assert not any("calcity" in split for split in citynav_bench({}).split_files)


def test_the_build_commands_default_to_the_bench_s_own_roots(monkeypatch, tmp_path):
    from benchmarks.calcity.data_prep import bake_tiles, build_objects_json, build_rgbd_ortho

    monkeypatch.setenv("SEARCH_CALCITY_DATA", str(tmp_path / "root"))
    monkeypatch.setenv("SEARCH_CALCITY_TILES", str(tmp_path / "tiles"))
    seen = {}
    monkeypatch.setattr(
        build_objects_json, "build", lambda root, bundle: seen.setdefault("objects", root)
    )
    monkeypatch.setattr(
        build_rgbd_ortho,
        "build_block",
        lambda root, name, bbox, raster: seen.setdefault("ortho", root) and (None, None),
    )
    monkeypatch.setattr(
        bake_tiles, "bake_block", lambda bundle, tiles, name: seen.setdefault("tiles", tiles) and 0
    )
    build_objects_json.main(["--bundle", str(tmp_path)])
    build_rgbd_ortho.main(["--raster", str(tmp_path / "r.tif")])
    bake_tiles.main(["--bundle", str(tmp_path)])
    assert seen == {
        "objects": tmp_path / "root",
        "ortho": tmp_path / "root",
        "tiles": tmp_path / "tiles",
    }
