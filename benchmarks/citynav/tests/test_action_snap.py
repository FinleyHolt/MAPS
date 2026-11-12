"""The CityNav six-action snap: which actions a hop decomposes into, and in
what order (turn, then forward, then altitude)."""

from __future__ import annotations

import math

import pytest

from benchmarks.citynav.action_snap import Pose, apply_action, snap_to_actions

START = Pose(0.0, 0.0, 50.0, 0.0)  # facing north


def _snap(x: float, y: float, alt: float = 50.0):
    return snap_to_actions(START, Pose(x, y, alt, 0.0))


def test_an_aligned_hop_is_forward_steps():
    actions, reached = _snap(0.0, 30.0)
    assert [a.value for a in actions] == ["move_forward"] * 6
    assert reached.y == pytest.approx(30.0, abs=1e-6)
    assert reached.altitude_m == pytest.approx(50.0)


def test_a_hop_off_heading_turns_first():
    actions, _ = _snap(30.0, 0.0)  # due east
    seq = [a.value for a in actions]
    assert seq[:3] == ["turn_right"] * 3
    assert set(seq[3:]) == {"move_forward"}


def test_an_altitude_change_in_place_is_altitude_steps():
    actions, reached = _snap(0.0, 0.0, alt=44.0)
    assert [a.value for a in actions] == ["descend"] * 3
    assert reached.altitude_m == pytest.approx(44.0)


def test_a_hop_already_reached_is_no_actions():
    assert _snap(0.0, 0.0) == ([], START)


def test_the_drone_descends_only_over_the_target():
    """Every turn precedes every forward step, and every forward step every
    descend: the drone never descends in place 35 m from its target."""
    seq = [a.value for a in _snap(35.0, 0.0, alt=30.0)[0]]
    turns = [i for i, a in enumerate(seq) if a in ("turn_right", "turn_left")]
    forwards = [i for i, a in enumerate(seq) if a == "move_forward"]
    descends = [i for i, a in enumerate(seq) if a == "descend"]
    assert turns and forwards and descends and "ascend" not in seq
    assert max(turns) < min(forwards) and max(forwards) < min(descends)


def test_replaying_the_actions_reproduces_the_reached_pose():
    actions, reached = snap_to_actions(Pose(5.0, 10.0, 52.0, 30.0), Pose(40.0, 60.0, 50.0, 0.0))
    pose = Pose(5.0, 10.0, 52.0, 30.0)
    for a in actions:
        pose = apply_action(pose, a)
    assert all(
        math.isclose(p, r, abs_tol=1e-9)
        for p, r in zip(
            (pose.x, pose.y, pose.altitude_m, pose.heading_deg),
            (reached.x, reached.y, reached.altitude_m, reached.heading_deg),
        )
    )
