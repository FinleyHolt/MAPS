#!/usr/bin/env python3
"""The reviewed Cal City episodes: corpus rows whose target shape was reviewed.

``ground_truth.json`` lists the position of each reviewed target,
``<map>::<x>:<y>``. A corpus row is reviewed when its ``target_positions[0]``
lies within ``TOL_M`` of a listed position on its own map. The list feeds
``bench_fullset --episode-allowlist``, which applies it before the seeded
shuffle::

    python -m benchmarks.calcity.data_prep.reviewed > reviewed_episodes.txt
    python -m benchmarks.calcity.bench_fullset --episode-allowlist reviewed_episodes.txt
"""

from __future__ import annotations

import json
import sys

import numpy as np

from benchmarks.calcity.paths import PACKAGE_ROOT, split_path
from benchmarks.citynav.adapter import parse_episode

GROUND_TRUTH = PACKAGE_ROOT / "data_prep" / "authoring" / "ground_truth.json"
# The positions are rounded to 0.1 m, so a match sits within ~0.07 m.
TOL_M = 0.2


def reviewed_episode_ids() -> list[str]:
    """Episode ids of the corpus rows that match a reviewed shape, in corpus order."""
    rows = json.loads(split_path("calcity").read_text())
    keys: dict[str, list[tuple[float, float]]] = {}
    for key in json.loads(GROUND_TRUTH.read_text()):
        map_name, _, xy = key.partition("::")
        x, _, y = xy.partition(":")
        keys.setdefault(map_name, []).append((float(x), float(y)))
    shapes = {m: np.asarray(pts) for m, pts in keys.items()}
    out = []
    for i, row in enumerate(rows):
        episode = parse_episode(row, row_index=i)
        pts = shapes.get(episode.map_name)
        if pts is None:
            continue
        target = np.asarray(row["target_positions"][0][:2], dtype=float)
        if np.min(np.hypot(*(pts - target).T)) <= TOL_M:
            out.append(episode.episode_id)
    return out


def main() -> int:
    sys.stdout.write("".join(f"{eid}\n" for eid in reviewed_episode_ids()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
