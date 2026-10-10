"""Seeded uniform spawn: the start pose when the config says ``spawn: uniform``.

The dataset's recorded start pose (``spawn: episode``) pins CityNav's
50-350 m start regime. A larger world (Cal City blocks run ~3.2 x 2-4 km)
wants the harder question — find the target from anywhere — so ``uniform``
replaces the start with a point drawn uniformly over the tile bounds,
deterministic per ``(spawn_seed, episode_id)``: one seed reproduces a run
exactly, two seeds give independent spawn sets: a hash fraction, with no
RNG state to share across workers.
"""

from __future__ import annotations

import hashlib

_HASH_DENOM = float(0xFFFFFFFFFFFFFFFF)


def uniform_spawn_xy(
    bounds_xyxy: tuple[float, float, float, float],
    spawn_seed: int,
    episode_id: str,
) -> tuple[float, float]:
    """A uniform in-bounds point, deterministic per (seed, episode)."""
    x_min, y_min, x_max, y_max = (float(v) for v in bounds_xyxy)
    digest = hashlib.sha256(
        f"spawn:{int(spawn_seed)}:{episode_id}".encode()
    ).hexdigest()
    fx = int(digest[:16], 16) / _HASH_DENOM
    fy = int(digest[16:32], 16) / _HASH_DENOM
    return (x_min + fx * (x_max - x_min), y_min + fy * (y_max - y_min))
