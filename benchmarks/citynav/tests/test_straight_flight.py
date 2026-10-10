"""Straight-line transit: the floor the constrained airframes are read against.

What has to hold is arithmetic — a leg of known length at a known speed takes a
known number of ticks and covers exactly that distance — plus the two liveness
properties the replay depends on for ANY model: an exhausted route holds
position instead of ending the stream, and a leg always terminates.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.citynav.flight_model import (
    DiscreteFlightSimulator,
    FlightModelConfig,
    resolve_flight_model,
)
from benchmarks.citynav.straight_flight import StraightFlightSimulator

# 10 m/s at 5 Hz = 2 m per tick, so every distance below divides evenly and the
# expected tick counts are exact rather than approximate.
CFG = FlightModelConfig(kind="straight", speed_mps=10.0, camera_hz=5.0)


def _sim(start=(0.0, 0.0, 0.0), alt=50.0):
    return StraightFlightSimulator(CFG, start, start_altitude_m=alt)


def _fly(sim, target, *, altitude_m=None, cap=10_000):
    poses = []
    sim.set_next_target(target, altitude_m=altitude_m)
    while sim.has_pending_target() and len(poses) < cap:
        poses.append(next(sim))
    assert len(poses) < cap, "leg did not terminate"
    return poses


def test_a_leg_takes_the_ticks_the_speed_says():
    """100 m at 2 m per tick = 50 ticks, landing ON the target."""
    sim = _sim()
    poses = _fly(sim, (0.0, 100.0))
    assert len(poses) == 50
    last = poses[-1]
    assert math.isclose(last.y, 100.0, abs_tol=1e-9)


def test_the_last_partial_tick_is_flown_so_the_commit_is_reached():
    """The commit leg sets pred_xy. Stopping at "within one tick" left it at
    the previous pose -- and a commit closer than one tick never moved the
    drone at all, scoring the START position."""
    sim = _sim()
    poses = _fly(sim, (0.0, 101.0))  # not a whole number of 2 m ticks
    assert math.isclose(poses[-1].y, 101.0, abs_tol=1e-9)

    near = _sim()
    poses = _fly(near, (0.0, 0.5))  # a quarter of one tick away
    assert len(poses) == 1
    assert math.isclose(poses[-1].y, 0.5, abs_tol=1e-9)


def test_the_path_flown_is_the_straight_line_length():
    """No quantization detour: what it covers is exactly what a ruler says."""
    sim = _sim()
    poses = _fly(sim, (60.0, 80.0))  # 100 m diagonal
    pts = [(0.0, 0.0)] + [(p.x, p.y) for p in poses]
    flown = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    assert math.isclose(flown, 100.0, abs_tol=1e-9)
    # and it went in a straight line to get there, not around
    assert math.isclose(poses[-1].x / poses[-1].y, 60.0 / 80.0, rel_tol=1e-9)


def test_heading_points_at_the_target_immediately():
    """No turn actions: a straight-flying airframe is already pointed."""
    sim = _sim(start=(0.0, 0.0, 180.0))
    poses = _fly(sim, (100.0, 0.0))  # due east
    assert math.isclose(poses[0].heading_deg, 90.0, abs_tol=1e-9)


def test_altitude_arrives_when_the_traverse_does():
    """A climb spread over the leg, not a vertical hop: this airframe has
    nowhere to hover while it changes height."""
    sim = _sim(alt=50.0)
    poses = _fly(sim, (0.0, 100.0), altitude_m=90.0)
    alts = [p.altitude_m for p in poses]
    assert alts == sorted(alts), "altitude must ramp monotonically"
    assert alts[0] > 50.0 and alts[-1] <= 90.0
    assert math.isclose(alts[-1], 90.0, abs_tol=2.0)
    # Mid-leg it is genuinely mid-climb, not already arrived.
    assert 50.0 < alts[len(alts) // 2] < 90.0


def test_a_descent_in_place_still_completes():
    """Zero horizontal distance, 20 m to lose: the leg must end, not spin."""
    sim = _sim(alt=50.0)
    poses = _fly(sim, (0.0, 0.0), altitude_m=30.0)
    assert poses, "a pure altitude change must still emit a tick"
    assert math.isclose(poses[-1].altitude_m, 30.0, abs_tol=1e-9)
    assert not sim.has_pending_target()


def test_an_exhausted_route_holds_position_instead_of_ending():
    """The replay's drain loop reads has_pending_target(); the stream itself
    must never raise StopIteration mid-episode the way a finished queue could."""
    sim = _sim()
    _fly(sim, (0.0, 20.0))
    assert not sim.has_pending_target()
    held = [next(sim) for _ in range(5)]
    assert len({(round(p.x, 9), round(p.y, 9)) for p in held}) == 1


def test_ticks_are_stamped_at_the_camera_rate():
    sim = _sim()
    poses = _fly(sim, (0.0, 40.0))
    assert [p.frame_idx for p in poses] == list(range(len(poses)))
    assert math.isclose(poses[3].t_s, 3 * 0.2, abs_tol=1e-9)


def test_no_citynav_action_name_is_emitted():
    """A continuous tick is not one of the six primitives. Naming it would
    make a log that cannot be submitted look like one that can."""
    sim = _sim()
    poses = _fly(sim, (0.0, 40.0))
    assert {p.action for p in poses} == {None}


def test_the_tolerance_is_not_a_tick_and_the_step_is():
    """Two different questions the replay asks, and conflating them biased
    every commit. ``advance_per_tick_m`` sizes the budget reserve, so it is
    the tick. ``arrival_tol_m`` is what the replay aims PAST the commit by to
    centre a short-stopping approach -- and this model does not stop short,
    so it must be ~0. A whole tick here put every straight landing PAST the
    target (5.9 m mean at the C-12's 11.3 m tick)."""
    sim = _sim()
    assert sim.advance_per_tick_m == 2.0
    assert sim.arrival_tol_m < 1e-3
    # and it is genuinely tighter than the discrete walk's half-step
    assert sim.arrival_tol_m < DiscreteFlightSimulator.arrival_tol_m


def test_path_length_does_not_depend_on_the_speed():
    """The claim configs/calcity_straight.yaml makes: speed sets tick count
    and time of flight, NOT the route. It only holds because the leg lands
    exactly -- while the model stopped a tick short, the endpoint (and so the
    path, and so pred_xy) moved with the speed."""
    lengths = set()
    for speed in (5.0, 10.0, 56.6, 120.0):
        cfg = FlightModelConfig(kind="straight", speed_mps=speed, camera_hz=5.0)
        sim = StraightFlightSimulator(cfg, (0.0, 0.0, 0.0), start_altitude_m=50.0)
        poses = _fly(sim, (600.0, 800.0))  # 1000 m diagonal
        pts = [(0.0, 0.0)] + [(p.x, p.y) for p in poses]
        lengths.add(round(sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)), 6))
    assert lengths == {1000.0}


def test_the_registry_builds_it_from_config():
    sim = resolve_flight_model(CFG, start_pose=(1.0, 2.0, 3.0), start_altitude_m=44.0)
    assert isinstance(sim, StraightFlightSimulator)
    assert sim.advance_per_tick_m == 2.0


def test_a_speed_the_kind_cannot_use_is_refused_not_ignored():
    """An intrinsic sitting unread in a config reads as a setting in force."""
    with pytest.raises(ValueError, match="no speed_mps"):
        FlightModelConfig(kind="discrete6", speed_mps=6.0)


def test_straight_without_a_speed_is_refused():
    with pytest.raises(ValueError, match="speed_mps"):
        FlightModelConfig(kind="straight")
    with pytest.raises(ValueError, match="speed_mps"):
        FlightModelConfig(kind="straight", speed_mps=0.0)


def test_straight_is_shorter_than_the_discrete_walk_over_the_same_stops():
    """The reason the arm exists, as an assertion: the discrete walk pays for
    30 deg heading quantization and 5 m steps, and the straight line does not.
    A regression that made straight LONGER would otherwise read as a result."""
    stops = [(140.0, 30.0), (60.0, 160.0), (-50.0, 40.0)]

    def _walk(cfg):
        sim = resolve_flight_model(cfg, start_pose=(0.0, 0.0, 0.0), start_altitude_m=50.0)
        pts = [(0.0, 0.0)]
        for stop in stops:
            for _ in range(4000):
                sim.set_next_target(stop)
                if not sim.has_pending_target():
                    break
                while sim.has_pending_target():
                    fp = next(sim)
                    pts.append((fp.x, fp.y))
                if math.dist(pts[-1], stop) <= sim.arrival_tol_m:
                    break
        return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))

    straight = _walk(CFG)
    discrete = _walk(FlightModelConfig())
    ideal = sum(
        math.dist(a, b) for a, b in zip([(0.0, 0.0)] + stops[:-1], stops)
    )
    assert straight < discrete
    assert straight == pytest.approx(ideal, abs=1e-6)
