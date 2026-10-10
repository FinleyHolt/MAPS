"""Compliant-trajectory scoring contract.

Pins the two invariants the commit path depends on: the synthesized
discrete trajectory scores its FINAL POSE (pred_xy == final_pos_xy — no
teleport credit for the committed centroid), and the citynav-native
scorer keeps final_pos_xy as the canonical scoring basis while carrying
pred diagnostics.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from benchmarks.citynav.flight_model import FlightModelConfig


def test_compliant_trajectory_records_final_pose_prediction_xy(
    tmp_path, monkeypatch
):
    from benchmarks.citynav.run_benchmark import (
        RunConfig,
        synthesize_compliant_trajectory,
    )

    class _State:
        def __init__(self):
            self.pose_xy = (0.0, 0.0)
            self.altitude_m = 50.0
            self.heading_deg = 0.0
            self.visited_xy = []
            self.belief_snapshot = SimpleNamespace(
                confirmed_target_xy=(100.0, 0.0),
            )

        def record_visit(self):
            self.visited_xy.append(self.pose_xy)

    cfg = RunConfig()
    cfg.flight_model = FlightModelConfig()
    cfg.max_steps = 1
    cfg.altitude_m = 50.0
    cfg.descend_to_land = False
    cfg.landing_altitude_m = 5.0
    state = _State()
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))

    synthesize_compliant_trajectory(
        cfg=cfg,
        state=state,
        episode=SimpleNamespace(episode_id="unit::0"),
        feature_tour=SimpleNamespace(viewpoints=[]),
        start_pose_xyh=(0.0, 0.0, 0.0),
        start_alt=50.0,
    )

    traj = json.loads((tmp_path / "run" / "compliant_trajectory.json").read_text())

    # One discrete step of budget cannot reach the committed centroid at
    # (100, 0): the honest final pose is what gets scored.
    assert traj["pred_xy"] == traj["final_pos_xy"]
    assert traj["final_pos_xy"] != [100.0, 0.0]


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
