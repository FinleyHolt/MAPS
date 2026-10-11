"""The citynav-native scorer scores the final pose and carries the prediction as diagnostics."""

from __future__ import annotations


def test_citynav_native_protocol_keeps_final_pose_as_canonical_score() -> None:
    from benchmarks.citynav.eval import score_goal_prediction_citynav_native

    scores = score_goal_prediction_citynav_native(
        final_pos_xy=(0.0, 0.0),
        pred_xy=(100.0, 0.0),
        target_xy=(100.0, 0.0),
        pos_trajectory=[(0.0, 0.0)],
        pred_trajectory=[(100.0, 0.0)],
        success_dist_m=20.0,
    )

    assert scores["scoring_basis"] == "final_pos_xy"
    assert scores["NE"] == 100.0
    assert scores["SR"] == 0
    assert scores["ne_pred_to_goal"] == 0.0
    assert scores["sr_pred_to_goal"] == 1
    assert scores["ne_pos_to_goal"] == 100.0
    assert scores["sr_pos_to_goal"] == 0


def _replay(commit_xy, max_steps=60):
    from types import SimpleNamespace

    from benchmarks.citynav.episode import EpisodeState
    from benchmarks.citynav.run_benchmark.trajectory import synthesize_compliant_trajectory
    from maps.search.parse_compile.tour_state import FeatureTour, TourViewpoint

    state = EpisodeState(pose_xy=(0.0, 0.0), altitude_m=50.0, graph=None)
    state.belief_snapshot = SimpleNamespace(confirmed_target_xy=commit_xy)
    tour = FeatureTour()
    tour.viewpoints.append(TourViewpoint(fid="far", xy=(0.0, 200.0), altitude_m=50.0))
    steps = synthesize_compliant_trajectory(
        cfg=SimpleNamespace(max_steps=max_steps, altitude_m=50.0), state=state, episode=None,
        feature_tour=tour, start_pose_xyh=(0.0, 0.0, 0.0), start_alt=50.0)
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
