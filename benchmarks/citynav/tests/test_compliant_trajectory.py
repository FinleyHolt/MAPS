"""The discrete replay reaches the commit (the last flight) within the action
budget, dropping flown stops before it starves the commit leg."""

from __future__ import annotations


def _replay(commit_xy, max_steps=60):
    from types import SimpleNamespace

    from benchmarks.citynav.run_benchmark.episode_runner import EpisodeState
    from benchmarks.citynav.run_benchmark.trajectory import synthesize_compliant_trajectory

    state = EpisodeState(pose_xy=(0.0, 0.0), altitude_m=50.0)
    flights = [{"target": "far", "resolved_xy": [0.0, 200.0], "altitude_m": 50.0},
               {"target": "commit", "resolved_xy": list(commit_xy), "altitude_m": 50.0}]
    steps = synthesize_compliant_trajectory(
        cfg=SimpleNamespace(max_steps=max_steps), state=state,
        flights=flights, start_pose_xyh=(0.0, 0.0, 0.0), start_alt=50.0)
    return steps, state


def test_a_commit_the_tour_starves_is_reached_by_dropping_the_tour() -> None:
    import math

    from benchmarks.citynav.action_snap import ARRIVAL_BOUND_M

    steps, state = _replay((0.0, -150.0))
    assert steps <= 60
    assert math.dist(state.pose_xy, (0.0, -150.0)) <= ARRIVAL_BOUND_M
    assert max(p[1] for p in state.visited_xy) <= 0.0, "the replanned replay never flies the tour"


def test_a_commit_beyond_the_whole_budget_ends_short_at_the_budget() -> None:
    steps, state = _replay((0.0, -1000.0))
    assert steps == 60 and state.pose_xy[1] > -1000.0
