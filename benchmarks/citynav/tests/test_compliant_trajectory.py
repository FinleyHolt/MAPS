"""The discrete replay reaches the commit (the last flight) within the action
budget, dropping flown stops before it starves the commit leg."""

from __future__ import annotations


def _replay(commit_xy, max_steps=60):
    from types import SimpleNamespace

    from benchmarks.citynav.run_benchmark.trajectory import synthesize_compliant_trajectory

    flights = [
        {"target": "far", "resolved_xy": [0.0, 200.0], "altitude_m": 50.0},
        {"target": "commit", "resolved_xy": list(commit_xy), "altitude_m": 50.0},
    ]
    return synthesize_compliant_trajectory(
        cfg=SimpleNamespace(max_steps=max_steps),
        flights=flights,
        start_pose_xyh=(0.0, 0.0, 0.0),
        start_alt=50.0,
    )


def test_a_commit_the_tour_starves_is_reached_by_dropping_the_tour() -> None:
    import math

    from benchmarks.citynav.action_snap import ARRIVAL_BOUND_M

    visited = _replay((0.0, -150.0))
    assert len(visited) <= 60
    assert math.dist(visited[-1], (0.0, -150.0)) <= ARRIVAL_BOUND_M
    assert max(p[1] for p in visited) <= 0.0, "the replanned replay never flies the tour"


def test_a_commit_beyond_the_whole_budget_ends_short_at_the_budget() -> None:
    visited = _replay((0.0, -1000.0))
    assert len(visited) == 60 and visited[-1][1] > -1000.0
