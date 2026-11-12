"""CityNav six-action snap for the benchmark step loop.

CityNav (Lee et al., arxiv 2406.14240v3, Sec. 3.1 / 4.2) scores
trajectories on a fixed discrete action set: ``move_forward 5 m``,
``turn_left 30 deg``, ``turn_right 30 deg``, ``ascend 2 m``,
``descend 2 m``, ``stop``. The compiler emits continuous viewpoints;
this module decomposes each continuous hop into the closest
CityNav-action sequence so the trajectory log is submittable. Consumed
by the replay (``run_benchmark/trajectory.py``).

The decomposition is greedy and deterministic: rotate toward bearing,
advance forward to the target XY (turning again in cruise only once the
target sits on the next 30-degree heading), then step altitude to the
target height. Altitude is the last phase so the drone never descends until
it is over the target XY (the standard quadcopter waypoint pattern).
The achieved pose may differ from the requested end pose when the
discrete action set cannot reach it exactly or the sequence hits
:data:`MAX_ACTIONS`; in that case the next hop re-plans from the achieved
pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class CityNavAction(str, Enum):
    """CityNav's movement actions, the primitives the trajectory log holds."""

    FORWARD = "move_forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    ASCEND = "ascend"
    DESCEND = "descend"


@dataclass(frozen=True)
class Pose:
    """4-DOF pose: planar position, altitude, heading (deg, 0 = north, CW)."""

    x: float
    y: float
    altitude_m: float
    heading_deg: float


# Per-action deltas defined by the CityNav contract.
FORWARD_STEP_M: float = 5.0
TURN_STEP_DEG: float = 30.0
ALTITUDE_STEP_M: float = 2.0

# How close to a target the discrete walk can actually stop: the step is 5 m
# and headings quantize to 30 deg, so it lands within about half a step of the
# point it aimed at. Named because everything that asks "did this leg arrive?"
# or "how far past the target should it aim?" has to read the FLYING model's
# tolerance -- a literal 2.5 copied into the replay is the discrete model's
# number silently applied to a continuous one.
SNAP_XY_TOL_M: float = 2.5


def _wrap_signed(angle_deg: float) -> float:
    """Wrap an angle into (-180, 180] degrees."""
    a = (angle_deg + 180.0) % 360.0 - 180.0
    return a + 360.0 if a <= -180.0 else a


def bearing_deg(from_xy: tuple[float, float], to_xy: tuple[float, float]) -> float:
    """Compass bearing from ``from_xy`` to ``to_xy`` (0 = north, CW).

    The one place this build states the convention, so every flight model
    points at a target the same way.
    """
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    return math.degrees(math.atan2(dx, dy))


def apply_action(pose: Pose, action: CityNavAction) -> Pose:
    """Advance a pose by one CityNav action."""
    if action is CityNavAction.TURN_LEFT:
        return Pose(
            pose.x,
            pose.y,
            pose.altitude_m,
            _wrap_signed(pose.heading_deg - TURN_STEP_DEG),
        )
    if action is CityNavAction.TURN_RIGHT:
        return Pose(
            pose.x,
            pose.y,
            pose.altitude_m,
            _wrap_signed(pose.heading_deg + TURN_STEP_DEG),
        )
    if action is CityNavAction.ASCEND:
        return Pose(pose.x, pose.y, pose.altitude_m + ALTITUDE_STEP_M, pose.heading_deg)
    if action is CityNavAction.DESCEND:
        return Pose(pose.x, pose.y, pose.altitude_m - ALTITUDE_STEP_M, pose.heading_deg)
    # FORWARD
    rad = math.radians(pose.heading_deg)
    return Pose(
        pose.x + FORWARD_STEP_M * math.sin(rad),
        pose.y + FORWARD_STEP_M * math.cos(rad),
        pose.altitude_m,
        pose.heading_deg,
    )


#: Bearing error a leg aligns within before its first forward step, and on
#: its final approach (:data:`NEAR_M`): half the turn step, so the nearest
#: lattice heading is taken.
ALIGN_TOL_DEG: float = 15.0
#: Bearing error a leg in cruise tolerates: it turns only once the target
#: sits on the next 30-degree heading, as upstream's executor turns only past
#: 30 degrees (``gsamllavanav/teacher/algorithm/lookahead.py``). Turning at
#: half the step instead leaves no dead band, and the drone alternates turn
#: and forward down the whole leg.
CRUISE_TOL_DEG: float = 29.0
#: Distance to the target inside which a leg aligns at :data:`ALIGN_TOL_DEG`
#: again, so it ends within :data:`ARRIVAL_BOUND_M` of the target.
NEAR_M: float = 10.0
#: How close to a leg's target counts as arrived: within
#: :data:`SNAP_XY_TOL_M`, or at the action lattice's closest point to a
#: target no 30-degree heading aligns with, which the final approach leaves
#: within ``SNAP_XY_TOL_M / cos(ALIGN_TOL_DEG)``.
ARRIVAL_BOUND_M: float = SNAP_XY_TOL_M / math.cos(math.radians(ALIGN_TOL_DEG))


#: Altitude error a leg leaves open.
SNAP_ALT_TOL_M: float = 1.0
#: A runaway guard that never binds: the runner's ``_drive_to`` loop holds
#: the per-episode budget.
MAX_ACTIONS: int = 1024


def snap_to_actions(start: Pose, end: Pose) -> tuple[list[CityNavAction], Pose]:
    """Greedy decomposition of one compiler hop into CityNav actions.

    Returns ``(actions, achieved_pose)``. ``achieved_pose`` may differ
    from ``end`` when the discrete action set cannot exactly reach it.
    """
    actions: list[CityNavAction] = []
    pose = start
    moved = False  # a forward step taken: the leg is in cruise

    def _emit(action: CityNavAction) -> bool:
        nonlocal pose
        if len(actions) >= MAX_ACTIONS:
            return False
        actions.append(action)
        pose = apply_action(pose, action)
        return True

    target_xy = (end.x, end.y)
    # Set when the leg has come as close as the action set reaches: a target
    # within one forward step at a bearing no 30-degree heading aligns with
    # would otherwise be paced around until the runaway guard.
    xy_stuck = False
    while len(actions) < MAX_ACTIONS:
        dx = target_xy[0] - pose.x
        dy = target_xy[1] - pose.y
        xy_err = math.hypot(dx, dy)
        alt_err = end.altitude_m - pose.altitude_m
        xy_open = xy_err > SNAP_XY_TOL_M and not xy_stuck
        bearing = bearing_deg((pose.x, pose.y), target_xy) if xy_open else pose.heading_deg
        heading_err = _wrap_signed(bearing - pose.heading_deg)

        if not xy_open and abs(alt_err) <= SNAP_ALT_TOL_M:
            break
        heading_tol_deg = CRUISE_TOL_DEG if moved and xy_err > NEAR_M else ALIGN_TOL_DEG
        if xy_open and abs(heading_err) > heading_tol_deg:
            if not _emit(CityNavAction.TURN_LEFT if heading_err < 0 else CityNavAction.TURN_RIGHT):
                break
            continue
        if xy_open:
            h = math.radians(pose.heading_deg)
            after = math.hypot(
                target_xy[0] - (pose.x + FORWARD_STEP_M * math.sin(h)),
                target_xy[1] - (pose.y + FORWARD_STEP_M * math.cos(h)),
            )
            if after >= xy_err:
                xy_stuck = True
                continue
            if not _emit(CityNavAction.FORWARD):
                break
            moved = True
            continue
        if abs(alt_err) > SNAP_ALT_TOL_M:
            if not _emit(CityNavAction.ASCEND if alt_err > 0 else CityNavAction.DESCEND):
                break
            continue
        break

    return actions, pose
