"""The Cal City bench as a value, built from an environment mapping.

:func:`calcity_bench` reads the ``SEARCH_CALCITY_*`` variables, with
``CALCITY_ROOT`` as the data root's fallback; they name every location. The build tree contains the
CityRefer-compatible object catalog and orthographic rasters; released
splits and tiles live with this package by default.

``SEARCH_CALCITY_RGBD_DIR`` names the imagery the bench crops: a directory
holding ``<map>_rgb.tif`` (the block's own map) beside ``<map>.tif`` (the
flat height plane on the same grid), built by
``benchmarks.calcity.data_prep.build_rgbd_ortho`` from the user's raster. It
defaults to ``data/rgbd_upstream`` under the data root. A second raster of
the block is a second directory, never an overwrite: the ortho loader
(``maps.perception.ortho.camera_model``) keys its mmap cache on the directory
alone and never checks the raster's freshness. The run stamps the resolved
directory on every row (``run_manifest.RUN_ENV_FLAGS``), so two runs over two
rasters read as two runs on one bake.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from benchmarks.bench import Bench

PACKAGE_ROOT = Path(__file__).resolve().parent

# ONE dataset over the one unified map.
SPLIT_FILES: dict[str, str] = {
    "calcity": "calcity.json",
}


def _env_path(environ: Mapping[str, str], name: str, default: Path) -> Path:
    value = environ.get(name)
    return Path(value if value else default).expanduser().resolve()


def bake_meta(tiles_dir: Path) -> dict:
    """The meta sidecar of the tile bake under ``tiles_dir``.

    Anything derived from the bake's frame, its bounds above all, comes from
    here rather than from a literal a re-bake cannot move.
    """
    path = Path(tiles_dir) / "calcity_meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"the Cal City bake under {tiles_dir} carries no calcity_meta.json. "
            "Build it with python -m benchmarks.calcity.data_prep.bake_tiles, "
            "or point SEARCH_CALCITY_TILES at a bake that has one."
        )
    return json.loads(path.read_text())


def _assert_available(bench: Bench) -> None:
    """Fail with the Cal City preparation recipe when runtime data is absent."""
    # The tile bake is checked by its meta file, not the directory: the
    # directory is gitignored, so a tree can carry it empty (an interrupted
    # bake, a copied checkout), and a missing bake must abort here. The
    # runner otherwise builds substitute graphs / features on the fly and
    # scores against the wrong priors with a normal-looking row.
    required = [bench.splits_dir, bench.objects_json, bench.tiles_dir / "calcity_meta.json"]
    required.extend(bench.split_path(split) for split in bench.split_files)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cal City data is incomplete under {bench.root}.\n"
            "Build it with:\n"
            "  python -m benchmarks.calcity.data_prep.build_objects_json\n"
            "  python -m benchmarks.calcity.data_prep.bake_tiles\n"
            "Override locations with SEARCH_CALCITY_DATA, "
            "SEARCH_CALCITY_SPLITS, and SEARCH_CALCITY_TILES.\n\n"
            f"Missing: {', '.join(str(path) for path in missing)}"
        )
    # The bake must speak the engine's world frame (x east, y north). A
    # bake authored to a y-south frame would score every N/S
    # relation mirrored with a normal-looking row, so it is refused here.
    frame = (bake_meta(bench.tiles_dir).get("coordinate_frame") or {}).get("y_axis")
    if frame != "north":
        raise RuntimeError(
            f"the Cal City bake under {bench.tiles_dir} declares coordinate_frame "
            f"y_axis={frame!r}; the engine's world frame is y-north. "
            "Re-bake with python -m benchmarks.calcity.data_prep.bake_tiles."
        )


def calcity_bench(environ: Mapping[str, str]) -> Bench:
    """The Cal City bench ``environ`` describes. It flies the CityNav engine.

    Its ``environ`` names the bench and its roots under the shared runner's
    variables too (``SEARCH_BENCHMARK_NAME`` and the four ``SEARCH_CITYNAV_*``):
    an attempt child selects Cal City by the first, and every run records all five.
    """
    root = _env_path(
        environ,
        "SEARCH_CALCITY_DATA",
        Path(environ.get("CALCITY_ROOT", Path.home() / "datasets" / "calcity")),
    )
    data = root / "data"
    splits = _env_path(environ, "SEARCH_CALCITY_SPLITS", PACKAGE_ROOT / "data")
    rgbd = _env_path(environ, "SEARCH_CALCITY_RGBD_DIR", data / "rgbd_upstream")
    tiles = _env_path(environ, "SEARCH_CALCITY_TILES", PACKAGE_ROOT / "tiles")
    return Bench(
        name="calcity",
        root=root,
        splits_dir=splits,
        objects_json=data / "cityrefer" / "objects.json",
        rgbd_dir=rgbd,
        tiles_dir=tiles,
        split_files=SPLIT_FILES,
        success_criterion="shape",
        default_split="calcity",
        engine_module="benchmarks.citynav.engine",
        availability=_assert_available,
        environ={
            **environ,
            "SEARCH_BENCHMARK_NAME": "calcity",
            "SEARCH_CITYNAV_DATA": str(root),
            "SEARCH_CITYNAV_SPLITS": str(splits),
            "SEARCH_CITYNAV_TILES": str(tiles),
            "SEARCH_CITYNAV_RGBD_DIR": str(rgbd),
        },
    )
