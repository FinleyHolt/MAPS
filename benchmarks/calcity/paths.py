"""Filesystem contract for the independent Cal City benchmark.

Only ``SEARCH_CALCITY_*`` variables are public.  The build tree contains the
CityRefer-compatible object catalog and orthographic rasters; released splits
and tiles live with this package by default.

``SEARCH_CALCITY_RGBD_DIR`` names the imagery the bench crops: a directory
holding ``<map>_rgb.tif`` (the block's own map) beside ``<map>.tif`` (the
flat height plane on the same grid), built by
``benchmarks.calcity.data_prep.build_rgbd_ortho`` from the user's raster. It
defaults to ``data/rgbd_upstream`` under the data root. A second raster of
the block is a second directory, never an overwrite: the ortho loader
(``maps.perception.ortho.camera_model``) keys its mmap cache on the directory
alone and never checks the raster's freshness. The run stamps the resolved
directory on every row (``run_manifest.ARM_ENV_FLAGS``), so two arms over two
rasters read as two arms on one bake.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value if value else default).expanduser().resolve()


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

CALCITY_ROOT = _env_path(
    "SEARCH_CALCITY_DATA",
    Path(os.environ.get("CALCITY_ROOT", Path.home() / "datasets" / "calcity")),
)
DATA_DIR = CALCITY_ROOT / "data"
SPLITS_DIR = _env_path("SEARCH_CALCITY_SPLITS", PACKAGE_ROOT / "data")
OBJECTS_JSON = DATA_DIR / "cityrefer" / "objects.json"
RGBD_DIR = DATA_DIR / "rgbd"
CALCITY_NATIVE_RGBD_DIR = _env_path(
    "SEARCH_CALCITY_RGBD_DIR", DATA_DIR / "rgbd_upstream"
)
TILES_DIR = _env_path("SEARCH_CALCITY_TILES", PACKAGE_ROOT / "tiles")

# WHAT COUNTS AS ARRIVING here: landing inside the target's own annotated
# footprint. Every row of the corpus resolves one -- a hand-drawn
# ``target_mask`` or a linked object's contour -- so ``success_radius_m`` is
# a fallback that fires on no episode. The twin of
# ``benchmarks.citynav.paths.SUCCESS_CRITERION``; the bridge rebinds it.
SUCCESS_CRITERION = "shape"

# ONE dataset over the one unified map. There used to be a second,
# ``calcity_precise``, differing from this one in a single field: the
# fallback radius, 15.0 against 20.0. That fallback fires when a row
# resolves no shape, and every one of the 1,023 rows resolves one, so the
# two datasets produced identical verdicts on identical episodes -- the
# "present but always empty" hazard at the size of a whole corpus. Under
# one criterion per bench there is nothing left for it to mean.
SPLIT_FILES: dict[str, str] = {
    "calcity": "calcity.json",
}

# The fallback success radius, for an episode with no annotated shape.
# Per-episode overrides live in the ground-truth sidecar; a shape supersedes
# the radius entirely, and on today's corpus every row has one. Mirrored in
# the benchmark.json manifest — validate.py pins the two together.
DEFAULT_SUCCESS_RADIUS_M: dict[str, float] = {
    "calcity": 20.0,
}


def bake_meta() -> dict:
    """The live tile bake's meta sidecar.

    Read at CALL time. ``TILES_DIR`` follows ``SEARCH_CALCITY_TILES``, so this
    names whatever bake is active; anything derived from the bake's frame --
    its bounds above all -- must come from here rather than from a literal a
    re-bake cannot move.
    """
    path = TILES_DIR / "calcity_meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"the Cal City bake under {TILES_DIR} carries no calcity_meta.json. "
            "Build it with python -m benchmarks.calcity.data_prep.bake_tiles, "
            "or point SEARCH_CALCITY_TILES at a bake that has one."
        )
    return json.loads(path.read_text())


def split_path(split: str) -> Path:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown Cal City split {split!r}. Known: {sorted(SPLIT_FILES)}")
    return SPLITS_DIR / SPLIT_FILES[split]


def rgbd_png(map_name: str) -> Path:
    return RGBD_DIR / f"{map_name}.png"


def rgbd_tiff(map_name: str) -> Path:
    return RGBD_DIR / f"{map_name}.tiff"


def tile_stem(map_name: str) -> str:
    return str(TILES_DIR / map_name)


def raw_ply(map_name: str) -> None:
    """Cal City is a 2D imagery world and has no SensatUrban PLY source."""
    del map_name
    return None


def assert_available() -> None:
    """Fail with the Cal City preparation recipe when runtime data is absent."""
    # The tile bake is checked by its meta file, not the directory: a
    # deployed tree carries the (gitkept) directory with no contents, and
    # a missing bake must abort HERE — the runner otherwise builds
    # substitute graphs / features on the fly and scores against the
    # wrong priors with a normal-looking row.
    required = [SPLITS_DIR, OBJECTS_JSON, TILES_DIR / "calcity_meta.json"]
    required.extend(split_path(split) for split in SPLIT_FILES)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cal City data is incomplete under {CALCITY_ROOT}.\n"
            "Build it with:\n"
            "  python -m benchmarks.calcity.data_prep.build_objects_json\n"
            "  python -m benchmarks.calcity.data_prep.bake_tiles\n"
            "Override locations with SEARCH_CALCITY_DATA, "
            "SEARCH_CALCITY_SPLITS, and SEARCH_CALCITY_TILES.\n\n"
            f"Missing: {', '.join(str(path) for path in missing)}"
        )
    # The bake must speak the engine's world frame (x east, y north). A
    # bake authored to the retired y-south frame would score every N/S
    # relation mirrored with a normal-looking row — refuse it loudly.
    frame = (bake_meta().get("coordinate_frame") or {}).get("y_axis")
    if frame != "north":
        raise RuntimeError(
            f"the Cal City bake under {TILES_DIR} declares coordinate_frame "
            f"y_axis={frame!r}; the engine's world frame is y-north. "
            "Re-bake with python -m benchmarks.calcity.data_prep.bake_tiles."
        )
