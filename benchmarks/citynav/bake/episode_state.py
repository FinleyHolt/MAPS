"""Prewrite a workdir's episode state so the dashboard has something to read.

A runner spends its first ~30 s standing up perception, and until the
recorder's first beat the workdir holds nothing. These sidecars are written
before the runner launches so the Scenario panel, the live banner and the
audit's identity fields are populated from the start:

  * ``inputs.json``: the episode's identity and frame — episode_id, split,
    tile_id, the block's world bounds and cell size.
  * ``episode.json``: the attempt's row at ``status: flying`` -- episode_id,
    map_name, description, start_xy, target_xy, bounds. The runner
    overwrites it as the attempt moves.
  * ``flight/.advance_state.json``: which episode is active.
  * ``flight/heartbeat.json``: ``status="starting"``, so the banner flips
    immediately rather than waiting on the recorder.

Every field comes from the split row, the block's ``_meta.json`` and the run
config. Nothing here reads a satellite: the review canvas serves tile-level
imagery by map name (``/api/tile/sat``), so a workdir needs no copy of it and
this write cannot fail for want of one. It used to — the writer resolved
``<map>_satellite.npz`` to embed a per-workdir PNG, its call site swallowed
the failure as non-fatal, and on any host without the raw PLYs that took
``bounds_world_xy``, ``cell_size_m`` and ``tile_id`` down with the image the
readers never asked for.


Standalone::

    python -m benchmarks.citynav.bake.episode_state \\
        --episode-id <map>::<ann_id> \\
        --split <split> \\
        --workdir <workdir> \\
        [--config <config-yaml>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from benchmarks.citynav.episodes import STATUS_FLYING, write_episode_json


_DEFAULT_CELL_SIZE_M = 5.0


def _write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` atomically, so a reader never sees a half file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _read_cell_size_m(config_path: Path | None) -> float:
    if config_path is None:
        return _DEFAULT_CELL_SIZE_M
    from benchmarks.citynav.run_benchmark.config import resolve_config  # noqa: PLC0415

    return float(resolve_config(config_path)[0]["cell_size_m"])


def prewrite_episode_state(
    *,
    workdir: Path,
    episode_id: str,
    split: str,
    config_path: Path | None = None,
    write_heartbeat: bool = True,
    split_file: Path | None = None,
) -> None:
    """Write the workdir's episode-state sidecars. See the module docstring.

    ``split_file`` is the split's own JSON when the caller owns it (the
    station's mission sets live beside the station, not under the bench's
    ``split_path``); ``split`` still names it on the sidecars.

    Raises ``ValueError`` when the episode is not in the split. Call sites
    swallow that — the runner still produces a valid workdir and valid
    metrics, the dashboard just shows blank panels until the recorder
    catches up.
    """
    from benchmarks.citynav.adapter import load_split, parse_episode
    from benchmarks.citynav.paths import split_path, tile_stem

    rows = load_split(Path(split_file) if split_file else split_path(split))
    matched = None
    for i, raw in enumerate(rows):
        try:
            ep = parse_episode(raw, row_index=i)
        except (KeyError, ValueError, TypeError):
            continue
        if ep.episode_id == episode_id or ep.episode_id.rsplit("#", 1)[0] == episode_id:
            matched = ep
            break
    if matched is None:
        raise ValueError(f"episode {episode_id!r} not found in split {split!r}")

    workdir = workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    block_name = matched.map_name
    meta_path = Path(f"{tile_stem(block_name)}_meta.json")
    if meta_path.exists():
        b = (json.loads(meta_path.read_text()).get("bounds") or {})
        bounds = [
            float(b.get("x_min", 0.0)),
            float(b.get("y_min", 0.0)),
            float(b.get("x_max", 0.0)),
            float(b.get("y_max", 0.0)),
        ]
    else:
        bounds = [0.0, 0.0, 0.0, 0.0]

    _write_json(
        workdir / "inputs.json",
        {
            "mode": "citynav",
            "tile_id": block_name,
            "episode_id": matched.episode_id,
            "split": split,
            "bounds_world_xy": bounds,
            "cell_size_m": _read_cell_size_m(config_path),
        },
    )

    write_episode_json(
        workdir,
        {
            "status": STATUS_FLYING,
            "episode_id": matched.episode_id,
            "map_name": matched.map_name,
            "description": matched.description,
            "start_xy": [float(matched.start_xy[0]), float(matched.start_xy[1])],
            "target_xy": [float(matched.target_xy[0]), float(matched.target_xy[1])],
            "bounds": bounds,
        },
    )
    flight_dir = workdir / "flight"
    flight_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        flight_dir / ".advance_state.json",
        {"episode_id": matched.episode_id, "split": split},
    )
    if write_heartbeat:
        _write_json(
            flight_dir / "heartbeat.json",
            {
                "ts": time.time(),
                "episode_id": matched.episode_id,
                "status": "starting",
                "step": 0,
            },
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prewrite a CityNav workdir's episode-state sidecars"
    )
    ap.add_argument("--episode-id", required=True, help="<map>::<ann_id>")
    ap.add_argument("--split", required=True)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    try:
        prewrite_episode_state(
            workdir=args.workdir,
            episode_id=args.episode_id,
            split=args.split,
            config_path=args.config,
        )
    except (OSError, ValueError) as exc:
        print(f"[episode-state] FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"[episode-state] wrote episode state into {args.workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
