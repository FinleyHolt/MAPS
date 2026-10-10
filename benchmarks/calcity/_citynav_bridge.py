"""Private compatibility boundary between Cal City and the CityNav engine.

The engine is currently implemented under :mod:`benchmarks.citynav`.  Cal City
keeps its public contract independent and installs this adapter before importing
that engine.  No Cal City split, path, or map geometry is registered in the
CityNav package itself.
"""

from __future__ import annotations

import os

from benchmarks.calcity import paths


def map_bounds() -> dict[str, tuple[float, float, float, float]]:
    """The one unified map's local runtime bounds, off the live bake's meta.

    Resolved at CALL time, never bound at import: ``paths.TILES_DIR`` follows
    ``SEARCH_CALCITY_TILES``, so a bake that moves the frame moves these with
    it. A literal here clips ``maps.perception``'s ortho to whichever frame
    the literal was written for while the corpus addresses the bake's own --
    targets outside the stale rectangle render as no-data, and nothing fails.
    A missing meta raises rather than falling back.
    """
    meta = paths.bake_meta()
    bounds = meta["bounds"]
    return {
        meta["map_name"]: (
            bounds["x_min"], bounds["y_min"], bounds["x_max"], bounds["y_max"],
        )
    }


def compat_env() -> dict[str, str]:
    """The public Cal City contract restated in the shared engine's variables.

    Enumerable rather than written straight into ``os.environ`` so a caller
    that must put the process back — a test, above all — can restore exactly
    the names this installs. These are process-global and outlive whoever set
    them: a leaked ``SEARCH_CITYNAV_TILES`` re-bakes
    ``benchmarks.citynav.paths.TILES_DIR`` on the next reload and every later
    reader sees Cal City's tiles.
    """
    return {
        "SEARCH_CITYNAV_DATA": str(paths.CALCITY_ROOT),
        "SEARCH_CITYNAV_SPLITS": str(paths.SPLITS_DIR),
        "SEARCH_CITYNAV_TILES": str(paths.TILES_DIR),
        "SEARCH_CITYNAV_RGBD_DIR": str(paths.CALCITY_NATIVE_RGBD_DIR),
    }


def install_compat_env() -> None:
    """Translate the public Cal City contract for shared child processes."""
    os.environ.update(compat_env())


# Every benchmark-specific public name of ``benchmarks.citynav.paths`` and its
# Cal City replacement. A shared-paths name missing from here (and from the
# test's exempt list of benchmark-independent names) means the engine silently
# reads CityNav data on a Cal City run —
# ``test_runtime_bridge.test_bridge_covers_every_shared_paths_name`` derives
# that check from the live module.
def _replacements() -> dict[str, object]:
    return {
        "CITYNAV_ROOT": paths.CALCITY_ROOT,
        "DATA_DIR": paths.DATA_DIR,
        "SPLITS_DIR": paths.SPLITS_DIR,
        "OBJECTS_JSON": paths.OBJECTS_JSON,
        "RGBD_DIR": paths.RGBD_DIR,
        "CITYNAV_NATIVE_RGBD_DIR": paths.CALCITY_NATIVE_RGBD_DIR,
        "TILES_DIR": paths.TILES_DIR,
        "SUCCESS_CRITERION": paths.SUCCESS_CRITERION,
        "SPLIT_FILES": paths.SPLIT_FILES,
        "split_path": paths.split_path,
        "rgbd_png": paths.rgbd_png,
        "rgbd_tiff": paths.rgbd_tiff,
        "tile_stem": paths.tile_stem,
        "raw_ply": paths.raw_ply,
        "assert_available": paths.assert_available,
    }


def activate(*, geometry: bool = True) -> None:
    """Point already-imported shared engine seams at the Cal City contract."""
    install_compat_env()

    from benchmarks.citynav import paths as shared_paths

    for name, value in _replacements().items():
        setattr(shared_paths, name, value)

    if not geometry:
        return

    from maps.perception.ortho import ground_level
    ground_level.GROUND_LEVEL.update({map_name: 0.0 for map_name in map_bounds()})
