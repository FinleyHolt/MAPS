from pathlib import Path

from benchmarks.calcity import paths


def test_split_contract_is_calcity_owned():
    assert paths.SPLIT_FILES == {"calcity": "calcity.json"}
    assert paths.split_path("calcity") == paths.SPLITS_DIR / "calcity.json"


def test_the_bench_declares_one_criterion_and_it_is_the_shape():
    """Every row of this corpus resolves a footprint, so the fallback
    radius fires on no episode."""
    assert paths.SUCCESS_CRITERION == "shape"


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
