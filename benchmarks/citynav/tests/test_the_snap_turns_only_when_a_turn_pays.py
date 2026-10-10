"""The discrete snap turns when the next 30-degree heading points closer, and
never paces around a target it cannot align with.

Upstream's executor (``gsamllavanav/teacher/algorithm/lookahead.py``) turns
only past 30 degrees of bearing error. A snap that turns at 15, half the turn
step, has no dead band: one forward step past the tolerance flips the error
to the other side and the drone alternates turn, forward, turn, forward to
the stop. A target within one forward step at a bearing no 30-degree heading
aligns with is as close as the lattice gets: a forward step that would not
close the distance ends the xy phase.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

from benchmarks.citynav.action_snap import (
    CityNavAction,
    Pose,
    apply_action,
    snap_to_actions,
)

#: The farthest a leg can end from its target: within the 2.5 m stop
#: tolerance along a heading up to 15 degrees off the bearing.
ARRIVAL_BOUND_M = 2.5 / math.cos(math.radians(15.0))
#: Inside this distance of its target a leg aligns at 15 degrees again.
NEAR_M = 10.0
TURNS = (CityNavAction.TURN_LEFT, CityNavAction.TURN_RIGHT)


def test_a_long_leg_ten_degrees_off_heading_turns_once():
    bearing = math.radians(10.0)
    end = Pose(200 * math.sin(bearing), 200 * math.cos(bearing), 120.0, 0.0)
    actions, _ = snap_to_actions(Pose(0.0, 0.0, 120.0, 0.0), end)
    turns = sum(a in TURNS for a in actions)
    assert turns <= 1, f"{turns} turns in {len(actions)} actions"


def _random_leg(rng: random.Random) -> tuple[Pose, Pose]:
    heading = rng.uniform(0, 360) if rng.random() < 0.5 else rng.choice([0, 30, 60, 90, 180, 270])
    start = Pose(rng.uniform(-50, 50), rng.uniform(-50, 50), 120.0, heading)
    end = Pose(rng.uniform(-300, 300), rng.uniform(-300, 300),
               120.0 + rng.choice([0, 0, 2, -4, 7.3]), 0.0)
    return start, end


def _cruise_turns(start: Pose, end: Pose, actions: list[CityNavAction]) -> int:
    """Turns taken after the leg's first forward step and farther than
    ``NEAR_M`` from its target: the leg in cruise."""
    pose, moved, turns = start, False, 0
    for action in actions:
        far = math.hypot(end.x - pose.x, end.y - pose.y) > NEAR_M
        turns += action in TURNS and moved and far
        moved = moved or action is CityNavAction.FORWARD
        pose = apply_action(pose, action)
    return turns


def test_a_leg_turns_at_most_once_in_cruise():
    rng = random.Random(0)
    for _ in range(3000):
        start, end = _random_leg(rng)
        actions, _ = snap_to_actions(start, end)
        assert _cruise_turns(start, end, actions) <= 1, (start, end)


def test_every_leg_ends_within_the_arrival_bound_without_pacing():
    rng = random.Random(0)
    for _ in range(3000):
        start, end = _random_leg(rng)
        actions, reached = snap_to_actions(start, end)
        assert len(actions) < 200, (start, end)
        assert math.hypot(end.x - reached.x, end.y - reached.y) <= ARRIVAL_BOUND_M + 1e-9
        assert abs(reached.altitude_m - end.altitude_m) <= 1.0 + 1e-9


def test_a_leg_never_orbits_its_target(monkeypatch, tmp_path):
    """A 162 m leg whose target no lattice heading aligns with: the replay
    flies it in under 100 actions and counts it reached."""
    from benchmarks.citynav.run_benchmark import (
        load_config,
        synthesize_compliant_trajectory,
    )

    class _State:
        pose_xy = (0.0, 0.0)
        altitude_m = 100.0
        heading_deg = 180.0
        belief_snapshot = None

        def __init__(self):
            self.visited_xy = []

        def record_visit(self):
            self.visited_xy.append(self.pose_xy)

    target = (154.77264176418146, -47.65705150149299)
    cfg = load_config("configs/citynav.yaml")
    cfg.max_steps = 100_000
    state = _State()
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    vp = SimpleNamespace(xy=target, altitude_m=100.0, heading_deg=None, fid="s0")
    steps = synthesize_compliant_trajectory(
        cfg=cfg,
        state=state,
        episode=SimpleNamespace(episode_id="unit::0"),
        feature_tour=SimpleNamespace(viewpoints=[vp]),
        start_pose_xyh=(0.0, 0.0, 180.0),
        start_alt=100.0,
    )
    assert steps < 100
    assert math.dist(state.pose_xy, target) <= ARRIVAL_BOUND_M
