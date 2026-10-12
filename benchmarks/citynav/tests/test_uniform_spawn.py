"""The uniform spawn is seeded, in-bounds, and actually uniform-ish.

``spawn: uniform`` replaces a dataset start with a deterministic draw per
(spawn_seed, episode_id). Reproducibility is the contract: the same seed
must reproduce a run's spawns exactly, and a different seed must give an
independent set — silently correlated spawns would make a "new" spawn set
re-measure the old one.
"""
from __future__ import annotations

from benchmarks.citynav.run_benchmark.episode_runner import uniform_spawn_xy

BOUNDS = (0.0, 0.0, 3227.6, 4078.7)
EIDS = [f"calcity::{i}" for i in range(200)]


def test_same_seed_reproduces_exactly():
    a = [uniform_spawn_xy(BOUNDS, 1234, e) for e in EIDS]
    b = [uniform_spawn_xy(BOUNDS, 1234, e) for e in EIDS]
    assert a == b


def test_every_draw_is_in_bounds():
    for e in EIDS:
        x, y = uniform_spawn_xy(BOUNDS, 1234, e)
        assert BOUNDS[0] <= x <= BOUNDS[2]
        assert BOUNDS[1] <= y <= BOUNDS[3]


def test_different_seeds_are_independent():
    a = [uniform_spawn_xy(BOUNDS, 1234, e) for e in EIDS]
    b = [uniform_spawn_xy(BOUNDS, 5678, e) for e in EIDS]
    assert all(pa != pb for pa, pb in zip(a, b))


def test_draws_spread_over_the_tile():
    """Vacuity floor for uniformity: every quadrant of the tile gets hits.
    A broken hash that clumped draws would pass the bounds check and still
    measure a corner of the map."""
    cx, cy = 0.5 * (BOUNDS[0] + BOUNDS[2]), 0.5 * (BOUNDS[1] + BOUNDS[3])
    quads = set()
    for e in EIDS:
        x, y = uniform_spawn_xy(BOUNDS, 1234, e)
        quads.add((x < cx, y < cy))
    assert len(quads) == 4
