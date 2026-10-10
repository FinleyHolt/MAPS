"""Launch one Cal City episode through the shared navigation engine."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from benchmarks.calcity._citynav_bridge import activate
from benchmarks.calcity.paths import (
    CALCITY_NATIVE_RGBD_DIR,
    TILES_DIR,
    tile_stem,
)

DEFAULT_CONFIG = "configs/calcity.yaml"
DEFAULT_RUN_ROOT = "output/calcity_runs"
SPLIT_PROBE_ORDER = ("calcity",)


def _rebuild_commands(*, map_name: str, missing_tiles: bool) -> list[tuple[str, list[str]]]:
    """What preflight may build on its own: the tile bake, from the OSM
    bundle (``CALCITY_BUNDLE``) into ``SEARCH_CALCITY_TILES``."""
    if not missing_tiles:
        return []
    return [
        (
            "Cal City tile artifacts",
            [sys.executable, "-m", "benchmarks.calcity.data_prep.bake_tiles",
             "--blocks", map_name],
        )
    ]


def preflight_artifacts(
    *,
    map_name: str,
    config_path: Path,
    repo_root: Path | None = None,
    auto_bake: bool = True,
) -> None:
    """Ensure the block's raster pair and its tile bake exist.

    The raster pair is ``<map>_rgb.tif`` (the block's own map, which the
    ortho loader streams into its cache) beside ``<map>.tif`` (the flat
    height plane on the same grid), both under ``SEARCH_CALCITY_RGBD_DIR``.
    The repo ships no imagery, so a missing pair is never built here: the
    failure names the raster and the command that lays the pair down from
    the user's own map.
    """
    repo = repo_root or Path(__file__).resolve().parents[2]
    stem = Path(tile_stem(map_name))
    tile_suffixes = ("_meta.json", "_features.json", "_graph.npz", "_nodes.json")
    rasters = [
        CALCITY_NATIVE_RGBD_DIR / f"{map_name}.tif",
        CALCITY_NATIVE_RGBD_DIR / f"{map_name}_rgb.tif",
    ]

    def missing_tiles() -> list[Path]:
        return [p for p in (Path(f"{stem}{s}") for s in tile_suffixes) if not p.exists()]

    if auto_bake:
        for label, command in _rebuild_commands(
            map_name=map_name, missing_tiles=bool(missing_tiles())
        ):
            print(f"[preflight] baking missing {label}...", file=sys.stderr)
            print(f"[preflight]   {' '.join(command)}", file=sys.stderr)
            subprocess.run(command, cwd=repo, check=False)

    tiles_gone = missing_tiles()
    rasters_gone = [p for p in rasters if not p.exists()]
    if not tiles_gone and not rasters_gone:
        print(
            f"[preflight] local Cal City assets ready (tiles={TILES_DIR}, "
            f"raster={rasters[1]})",
            file=sys.stderr,
        )
        return

    print(
        f"[preflight] FAIL: local Cal City assets missing for {map_name} "
        f"(config={config_path})",
        file=sys.stderr,
    )
    for path in (*tiles_gone, *rasters_gone):
        print(f"[preflight]   {path}", file=sys.stderr)
    if tiles_gone:
        print(
            "[preflight] bake the tiles: python -m benchmarks.calcity.data_prep.bake_tiles "
            "(OSM bundle at CALCITY_BUNDLE)",
            file=sys.stderr,
        )
    if rasters_gone:
        print(
            "[preflight] lay down the raster pair from the block's map: "
            "python -m benchmarks.calcity.data_prep.build_rgbd_ortho --raster <calcity_rgb.tif>",
            file=sys.stderr,
        )
    raise SystemExit(2)


def _configure_shared_launcher():
    """Install Cal City's contract and return the configured shared launcher."""
    os.environ["SEARCH_BENCHMARK_RUNNER_MODULE"] = "benchmarks.calcity.run_benchmark"
    os.environ["SEARCH_BENCHMARK_LAUNCH_MODULE"] = "benchmarks.calcity.launch"
    os.environ["SEARCH_BENCHMARK_NAME"] = "calcity"
    os.environ["SEARCH_BENCHMARK_LABEL"] = "Cal City"
    os.environ["SEARCH_BENCHMARK_EPISODE_EXAMPLE"] = "calcity::35"
    activate(geometry=False)

    from benchmarks.citynav import launch as shared

    shared.DEFAULT_CONFIG = DEFAULT_CONFIG
    shared.DEFAULT_RUN_ROOT = DEFAULT_RUN_ROOT
    shared.SPLIT_PROBE_ORDER = SPLIT_PROBE_ORDER
    shared.RUNNER_MODULE = "benchmarks.calcity.run_benchmark"
    shared.LAUNCH_MODULE = "benchmarks.calcity.launch"
    shared.BENCHMARK_LABEL = "Cal City"
    shared.EPISODE_EXAMPLE = "calcity::35"
    shared.preflight_artifacts = preflight_artifacts
    return shared


def main() -> int:
    shared = _configure_shared_launcher()
    return shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
