from pathlib import Path

from benchmarks.calcity import paths


def test_split_contract_is_calcity_owned():
    assert paths.SPLIT_FILES == {"calcity": "calcity.json"}
    assert paths.split_path("calcity") == paths.SPLITS_DIR / "calcity.json"


def test_default_release_layout_is_inside_calcity_package(monkeypatch):
    """The default falls inside the package when nothing overrides it.

    ``paths.TILES_DIR`` is a module-level constant baked in at import time,
    so reading it here would check whatever the ambient environment was on
    first import -- not the default. A session that legitimately sets
    ``SEARCH_CALCITY_TILES`` to a shared bake would fail that reading for
    saying nothing about a real regression. Call the same
    helper the constants are built from instead, with the overrides cleared.
    """
    monkeypatch.delenv("SEARCH_CALCITY_TILES", raising=False)
    monkeypatch.delenv("SEARCH_CALCITY_SPLITS", raising=False)
    package = Path(paths.__file__).resolve().parent
    assert paths._env_path("SEARCH_CALCITY_SPLITS", package / "data") == package / "data"
    assert paths._env_path("SEARCH_CALCITY_TILES", package / "tiles") == package / "tiles"


def test_citynav_registry_contains_no_calcity_splits():
    from benchmarks.citynav.paths import SPLIT_FILES as citynav_splits

    assert not any("calcity" in split for split in citynav_splits)


def test_the_build_commands_default_to_the_bench_s_own_roots(monkeypatch, tmp_path):
    from benchmarks.calcity import paths
    from benchmarks.calcity.data_prep import bake_tiles, build_objects_json, build_rgbd_ortho

    monkeypatch.setattr(paths, "CALCITY_ROOT", tmp_path / "root")
    monkeypatch.setattr(paths, "TILES_DIR", tmp_path / "tiles")
    seen = {}
    monkeypatch.setattr(build_objects_json, "build", lambda root, bundle: seen.setdefault("objects", root))
    monkeypatch.setattr(build_rgbd_ortho, "build_block",
                        lambda root, name, bbox, raster: seen.setdefault("ortho", root) and (None, None))
    monkeypatch.setattr(bake_tiles, "bake_block",
                        lambda bundle, tiles, name: seen.setdefault("tiles", tiles) and {"features": 0, "nodes": 0})
    build_objects_json.main(["--bundle", str(tmp_path)])
    build_rgbd_ortho.main(["--raster", str(tmp_path / "r.tif")])
    bake_tiles.main(["--bundle", str(tmp_path)])
    assert seen == {"objects": tmp_path / "root", "ortho": tmp_path / "root", "tiles": tmp_path / "tiles"}
