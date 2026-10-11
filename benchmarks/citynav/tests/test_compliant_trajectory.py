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
