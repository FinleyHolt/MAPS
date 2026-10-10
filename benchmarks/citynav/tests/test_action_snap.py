"""CityNav six-action snap integration tests.

Covers the benchmark-side surface in two slices:

1. ``DiscreteFlightSimulator`` yields one CityNav action per tick,
   updates pose deterministically.
2. The live-recorder JSON carries ``discrete_action`` and
   ``discrete_steps_emitted`` fields when the discrete flight model is
   active.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.citynav import action_snap as bench_snap
from benchmarks.citynav.flight_model import (
    DiscreteFlightSimulator,
    FlightModelConfig,
)
from benchmarks.citynav.live_recorder import LiveRecorder

ALLOWED_ACTIONS = {
    "move_forward",
    "turn_left",
    "turn_right",
    "ascend",
    "descend",
    "stop",
}


def test_action_snap_contract_constants():
    """The CityNav action-delta constants match the benchmark contract."""
    assert bench_snap.FORWARD_STEP_M == 5.0
    assert bench_snap.TURN_STEP_DEG == 30.0
    assert bench_snap.ALTITUDE_STEP_M == 2.0


def test_discrete_sim_emits_one_action_per_tick():
    cfg = FlightModelConfig(camera_hz=5.0)
    sim = DiscreteFlightSimulator(
        cfg=cfg,
        start_pose=(0.0, 0.0, 0.0),
        start_altitude_m=50.0,
    )
    sim.set_next_target((0.0, 30.0))
    poses = []
    for _ in range(20):
        try:
            poses.append(next(sim))
        except StopIteration:
            break

    actions = [p.action for p in poses if p.action != "stop"]
    assert actions, "snap should emit at least one action"
    assert all(a in ALLOWED_ACTIONS for a in actions)
    # 30 m due-north hop, drone already aligned: should be six forward
    # steps (5 m each) covering 30 m, then stops.
    forwards = [p for p in poses if p.action == "move_forward"]
    assert len(forwards) == 6
    final = poses[-1]
    assert final.y == pytest.approx(30.0, abs=1e-6)
    assert final.altitude_m == pytest.approx(50.0)


def test_discrete_sim_turn_then_forward():
    cfg = FlightModelConfig()
    sim = DiscreteFlightSimulator(
        cfg=cfg,
        start_pose=(0.0, 0.0, 0.0),
        start_altitude_m=50.0,
    )
    sim.set_next_target((30.0, 0.0))  # 30 m due east, drone faces north
    seq = []
    for _ in range(20):
        try:
            seq.append(next(sim).action)
        except StopIteration:
            break
    # Three turn_right 30 deg actions to face east (90 deg), then
    # forward steps until xy reaches the target.
    assert seq[:3] == ["turn_right", "turn_right", "turn_right"]
    assert "move_forward" in seq
    assert all(a in ALLOWED_ACTIONS for a in seq)


def test_discrete_sim_altitude_change_via_descent():
    cfg = FlightModelConfig()
    sim = DiscreteFlightSimulator(
        cfg=cfg,
        start_pose=(0.0, 0.0, 0.0),
        start_altitude_m=50.0,
    )
    sim.set_next_target((0.0, 0.0), altitude_m=44.0)
    seq = []
    while sim.has_pending_target():
        seq.append(next(sim).action)
    assert seq == ["descend", "descend", "descend"]
    final = sim._pose  # state after the queue drained
    assert final.altitude_m == pytest.approx(44.0)


def test_discrete_sim_hover_emits_stop():
    cfg = FlightModelConfig()
    sim = DiscreteFlightSimulator(
        cfg=cfg,
        start_pose=(0.0, 0.0, 0.0),
        start_altitude_m=50.0,
    )
    sim.set_next_target((0.0, 0.0))  # no movement needed
    for _ in range(3):
        pose = next(sim)
        assert pose.action == "stop"
        assert pose.x == 0.0
        assert pose.y == 0.0


def test_discrete_sim_determinism():
    cfg = FlightModelConfig()

    def collect():
        sim = DiscreteFlightSimulator(
            cfg=cfg,
            start_pose=(5.0, 10.0, 30.0),
            start_altitude_m=52.0,
        )
        sim.set_next_target((40.0, 60.0))
        out = []
        for _ in range(40):
            try:
                p = next(sim)
                out.append((p.x, p.y, p.altitude_m, p.heading_deg, p.action))
            except StopIteration:
                break
        return out

    assert collect() == collect()


def test_live_recorder_records_discrete_action(tmp_path: Path):
    rec = LiveRecorder(tmp_path, episode_id="ep0", split="val_seen")
    rec.write_step(
        step=0,
        pose_xy=(0.0, 0.0),
        altitude_m=50.0,
        heading_deg=0.0,
        visited_xy=[(0.0, 0.0)],
        fov_corners_xy=None,
        detections=[],
        compile_info=None,
        wall_t_monotonic=0.0,
        discrete_action="move_forward",
        discrete_steps_emitted=1,
    )
    rec.stop()
    payload = json.loads((tmp_path / "steps.jsonl").read_text())
    assert payload["discrete_action"] == "move_forward"
    assert payload["discrete_steps_emitted"] == 1


def test_live_recorder_back_compat_when_discrete_omitted(tmp_path: Path):
    """Legacy callers (continuous mode) get nulls without erroring."""
    rec = LiveRecorder(tmp_path, episode_id="ep0", split="val_seen")
    rec.write_step(
        step=0,
        pose_xy=(0.0, 0.0),
        altitude_m=50.0,
        heading_deg=0.0,
        visited_xy=[(0.0, 0.0)],
        fov_corners_xy=None,
        detections=[],
        compile_info=None,
        wall_t_monotonic=0.0,
    )
    rec.stop()
    payload = json.loads((tmp_path / "steps.jsonl").read_text())
    assert payload["discrete_action"] is None
    assert payload["discrete_steps_emitted"] == 0


def test_discrete_sim_translate_then_descend_ordering():
    """Regression: descend must wait until drone is over the target XY.

    Without the heading-XY-altitude phase order, the snap would consume the
    altitude delta first and the drone would descend in place 35 m from the
    locked target. Assert every move_forward precedes every descend.
    """
    cfg = FlightModelConfig()
    sim = DiscreteFlightSimulator(
        cfg=cfg,
        start_pose=(0.0, 0.0, 0.0),  # facing north
        start_altitude_m=50.0,
    )
    sim.set_next_target((35.0, 0.0), altitude_m=30.0)
    seq = []
    while sim.has_pending_target():
        seq.append(next(sim).action)
    assert "move_forward" in seq
    assert "descend" in seq
    assert "ascend" not in seq
    last_forward = max(i for i, a in enumerate(seq) if a == "move_forward")
    first_descend = min(i for i, a in enumerate(seq) if a == "descend")
    assert last_forward < first_descend
    # Heading turns must come before any forward step.
    turn_indices = [i for i, a in enumerate(seq) if a in ("turn_right", "turn_left")]
    first_forward = min(i for i, a in enumerate(seq) if a == "move_forward")
    assert turn_indices and max(turn_indices) < first_forward


def test_round_trip_simulated_actions_match_recorded_pose():
    """Re-simulating the action stream reproduces the simulator's pose."""
    cfg = FlightModelConfig()
    sim = DiscreteFlightSimulator(
        cfg=cfg,
        start_pose=(0.0, 0.0, 0.0),
        start_altitude_m=50.0,
    )
    sim.set_next_target((25.0, 25.0))
    actions = []
    poses = []
    for _ in range(40):
        try:
            p = next(sim)
            actions.append(p.action)
            poses.append((p.x, p.y, p.altitude_m, p.heading_deg))
        except StopIteration:
            break
    # Replay through bench_snap.apply_action and confirm final pose matches.
    pose = bench_snap.Pose(0.0, 0.0, 50.0, 0.0)
    for a in actions:
        pose = bench_snap.apply_action(pose, bench_snap.CityNavAction(a))
    assert math.isclose(pose.x, poses[-1][0], abs_tol=1e-9)
    assert math.isclose(pose.y, poses[-1][1], abs_tol=1e-9)
    assert math.isclose(pose.altitude_m, poses[-1][2], abs_tol=1e-9)
    assert math.isclose(pose.heading_deg, poses[-1][3], abs_tol=1e-9)
