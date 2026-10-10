"""Launch one Cal City episode through the shared navigation engine."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from benchmarks.calcity._citynav_bridge import activate
from benchmarks.calcity.paths import (
    CALCITY_NATIVE_RGBD_DIR,
    CALCITY_ROOT,
    TILES_DIR,
    tile_stem,
)

DEFAULT_CONFIG = "configs/calcity.yaml"
DEFAULT_RUN_ROOT = "output/calcity_runs"
SPLIT_PROBE_ORDER = ("calcity",)


def _rebuild_commands(
    *,
    map_name: str,
    repo: Path,
    missing_tiles: bool,
    missing_rasters: bool,
) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    if missing_rasters:
        commands.append(
            (
                "Cal City ortho rasters",
                [
                    sys.executable,
                    "-m",
                    "benchmarks.calcity.data_prep.build_rgbd_ortho",
                    "--root",
                    str(CALCITY_ROOT),
                    "--px",
                    "0.3",
                    "--blocks",
                    map_name,
                ],
            )
        )
    if missing_tiles:
        commands.append(
            (
                "Cal City tile artifacts",
                [str(repo / "benchmarks" / "calcity" / "data_prep" / "bake.sh"), map_name],
            )
        )
    return commands


def preflight_artifacts(
    *,
    map_name: str,
    config_path: Path,
    repo_root: Path | None = None,
    auto_bake: bool = True,
) -> None:
    """Ensure Cal City rasters and feature-graph artifacts exist."""
    repo = repo_root or Path(__file__).resolve().parents[2]
    stem = Path(tile_stem(map_name))
    tile_suffixes = (
        "_meta.json",
        "_features.json",
        "_graph.npz",
        "_nodes.json",
    )

    def missing() -> tuple[list[Path], list[Path]]:
        tiles = [Path(f"{stem}{suffix}") for suffix in tile_suffixes]
        rasters = [
            CALCITY_NATIVE_RGBD_DIR / f"{map_name}.tif",
            CALCITY_NATIVE_RGBD_DIR / f"{map_name}.png",
        ]
        return (
            [path for path in tiles if not path.exists()],
            [path for path in rasters if not path.exists()],
        )

    missing_tiles, missing_rasters = missing()
    if not missing_tiles and not missing_rasters:
        print(
            f"[preflight] local Cal City assets ready (tiles={TILES_DIR})",
            file=sys.stderr,
        )
        return

    if auto_bake:
        for label, command in _rebuild_commands(
            map_name=map_name,
            repo=repo,
            missing_tiles=bool(missing_tiles),
            missing_rasters=bool(missing_rasters),
        ):
            print(f"[preflight] baking missing {label}...", file=sys.stderr)
            print(f"[preflight]   {' '.join(command)}", file=sys.stderr)
            try:
                subprocess.run(command, cwd=repo, check=False)
            except FileNotFoundError as exc:
                print(f"[preflight] bake command failed: {exc}", file=sys.stderr)
        missing_tiles, missing_rasters = missing()
        if not missing_tiles and not missing_rasters:
            return

    print(
        f"[preflight] FAIL: local Cal City assets missing for {map_name} "
        f"(config={config_path})",
        file=sys.stderr,
    )
    for path in (*missing_tiles, *missing_rasters):
        print(f"[preflight]   {path}", file=sys.stderr)
    print(
        "[preflight] rebuild with benchmarks/calcity/data_prep/prepare.sh",
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
