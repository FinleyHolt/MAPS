import inspect
import json
import os

import pytest

from benchmarks.calcity import paths
from benchmarks.calcity._citynav_bridge import (
    _replacements,
    activate,
    compat_env,
    map_bounds,
)


@pytest.fixture
def synthetic_bake(tmp_path, monkeypatch):
    """A tiles dir whose meta declares a frame no literal in the tree carries.

    The geometry half of ``activate()`` is only worth asserting against a
    frame the assertion could not have guessed: read from the live bake, both
    sides of ``MAP_BOUNDS[map] == the meta`` move together and the test passes
    on a bridge that ignores the meta entirely. ``bake_meta`` reads
    ``TILES_DIR`` at call time, so patching the module global is enough.
    """
    meta = {
        "map_name": "calcity",
        "coordinate_frame": {"x_axis": "east", "y_axis": "north",
                             "origin": "southwest"},
        # Not the live bake's extent, and not a rounding of it: the numbers
        # here have to be ones the tree cannot supply, or a bridge that
        # ignored the meta entirely would satisfy the assertions below.
        "bounds": {"x_min": 0.0, "y_min": 0.0,
                   "x_max": 7311.5, "y_max": 4207.25},
    }
    (tmp_path / "calcity_meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(paths, "TILES_DIR", tmp_path)
    return meta


@pytest.fixture
def restores_compat_env():
    """Put the shared engine's environment back after ``activate()``.

    ``install_compat_env`` writes into the real ``os.environ``, which is
    process-global: left set, ``SEARCH_CITYNAV_TILES`` points at Cal City's
    bake for the rest of the pytest session, and the next
    ``importlib.reload(benchmarks.citynav.paths)`` — ``test_block_assets``
    does one — bakes it into ``TILES_DIR`` for every later reader. That is
    how a Cal City test came to fail ``test_ledger_reader``'s bake pin.
    """
    before = {name: os.environ.get(name) for name in compat_env()}
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture
def restores_shared_paths():
    """Put every name ``activate()`` rebinds back on ``citynav.paths``.

    The env half of the leak has its own fixture above; this is the module
    half, and it is derived from ``_replacements()`` rather than listed.
    A hand-written restore covers the names the author remembered:
    ``test_bake_sha256_follows_the_rebound_tiles_dir`` put back ``TILES_DIR``
    and left the other thirteen rebound for the rest of the session, which
    is how ``SUCCESS_CRITERION`` came to read "shape" inside a CityNav test
    that ran after it.
    """
    from benchmarks.citynav import paths as shared_paths

    names = list(_replacements())
    assert len(names) >= 15, "the bridge lost names -- this restore is derived"
    before = {name: getattr(shared_paths, name) for name in names}
    try:
        yield shared_paths
    finally:
        for name, value in before.items():
            setattr(shared_paths, name, value)


def test_bridge_installs_calcity_contract(
    restores_compat_env, restores_shared_paths, synthetic_bake
):
    from benchmarks.citynav import paths as shared_paths
    from maps.perception.ortho.ground_level import GROUND_LEVEL

    # The rebound names are put back by ``restores_shared_paths``, which
    # derives them from ``_replacements()``. A list here would be a second
    # copy of that dict, and the copy is what goes stale.
    try:
        activate()
        # The child-process half of the contract. Asserted here so the
        # fixture above cannot quietly restore a set of names that
        # ``install_compat_env`` has stopped writing.
        assert os.environ["SEARCH_CITYNAV_TILES"] == str(paths.TILES_DIR)
        assert os.environ["SEARCH_CITYNAV_SPLITS"] == str(paths.SPLITS_DIR)
        assert shared_paths.SPLIT_FILES == paths.SPLIT_FILES
        assert shared_paths.OBJECTS_JSON == paths.OBJECTS_JSON
        assert shared_paths.TILES_DIR == paths.TILES_DIR
        assert shared_paths.SUCCESS_CRITERION == "shape"
        assert shared_paths.split_path("calcity") == paths.split_path(
            "calcity"
        )
        # The map the BAKE declares, not a constant this module carries.
        assert GROUND_LEVEL[synthetic_bake["map_name"]] == 0.0
    finally:
        for map_name in (synthetic_bake["map_name"],):
            GROUND_LEVEL.pop(map_name, None)


def test_map_bounds_refuses_a_bake_with_no_meta(tmp_path, monkeypatch):
    """A silent fallback to a literal is the failure mode this replaces."""
    monkeypatch.setattr(paths, "TILES_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="calcity_meta.json"):
        map_bounds()


# Public shared-paths names the bridge deliberately does NOT rebind. Only a
# name with no benchmark-specific state belongs here — anything else silently
# reads CityNav data on a Cal City run.
_BENCHMARK_INDEPENDENT = {
    # The checkout root; the same tree runs both benchmarks.
    "REPO_ROOT",
    # Pure content validator: takes an already-loaded features payload and
    # checks its SHAPE (tags-only prior rows). Cal City's bake emits the same
    # shape (test_bake_tiles), so the guard applies to both benchmarks as-is.
    "assert_current_tile_schema",
    # The citynav bake NAME pin (card suffix == paths default,
    # test_perception_cards). Consumed only at import to build the
    # TILES_DIR default — which the bridge rebinds — and by that test;
    # no runtime path reads it, so a Cal City run cannot reach CityNav
    # data through it.
    "DEFAULT_TILE_BAKE",
    # Content hash of whatever bake TILES_DIR names, read from the module
    # global at CALL time. Rebinding it would bind the same function to the
    # same behaviour; it already follows the bridge, which
    # test_bake_sha256_follows_the_rebound_tiles_dir proves rather than
    # assumes — the distinction that matters is derived state, not no state.
    "bake_sha256",
}


def test_bridge_covers_every_shared_paths_name():
    """Derived coverage: a 15th public name added to ``benchmarks.citynav.paths``
    without a bridge entry fails HERE, not as CityNav data silently read on a
    Cal City run. Import artifacts (``Path``, ``annotations``) are recognized
    by origin; everything the module itself defines must be rebound or named
    benchmark-independent above."""
    from benchmarks.citynav import paths as shared_paths

    public = {
        n
        for n, v in vars(shared_paths).items()
        if not n.startswith("_")
        and not inspect.ismodule(v)
        and n not in ("Path", "annotations")  # import artifacts
    }
    rebound = set(_replacements())
    # Vacuity floor: the scan and the bridge both carry a real surface.
    assert len(public) >= 17, f"scan found only {sorted(public)}"
    assert len(rebound) >= 15
    uncovered = public - rebound - _BENCHMARK_INDEPENDENT
    assert not uncovered, (
        f"benchmarks.citynav.paths grew {sorted(uncovered)} and the bridge does "
        "not rebind them — a Cal City run would silently read CityNav data. "
        "Add them to _replacements() or, only if they carry no benchmark "
        "state, to _BENCHMARK_INDEPENDENT."
    )


def test_bake_sha256_follows_the_rebound_tiles_dir(
    tmp_path, monkeypatch, restores_compat_env, restores_shared_paths
):
    """``bake_sha256`` is in ``_BENCHMARK_INDEPENDENT`` because it reads
    ``TILES_DIR`` from the module global at CALL time, so the bridge's rebind
    already reaches it. That is a claim about behaviour, not about the name, so
    it is checked rather than asserted in a comment: a future rewrite that
    captured the default at import would keep the same signature, keep passing
    the coverage test above, and start hashing the CityNav bake on a Cal City
    run — stamping every result with the wrong bake identity.

    Takes both restore fixtures for the reason their docstrings give:
    ``activate()`` writes process-global state, and a Cal City test that
    leaves it set is what once broke ``test_ledger_reader``'s bake pin.
    This test used to put back ``TILES_DIR`` alone and leak the rest.
    """
    from benchmarks.citynav import paths as shared

    (tmp_path / "calcity_meta.json").write_text("{}")
    monkeypatch.setattr(paths, "TILES_DIR", tmp_path)
    activate(geometry=False)
    assert shared.bake_sha256() == shared.bake_sha256(paths.TILES_DIR)
    # Not vacuous: a bake that hashes to None either way would pass.
    assert shared.bake_sha256() is not None
