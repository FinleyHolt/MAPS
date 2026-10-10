"""CityNav six-action snap for the benchmark step loop.

CityNav (Lee et al., arxiv 2406.14240v3, Sec. 3.1 / 4.2) scores
trajectories on a fixed discrete action set: ``move_forward 5 m``,
``turn_left 30 deg``, ``turn_right 30 deg``, ``ascend 2 m``,
``descend 2 m``, ``stop``. The compiler emits continuous viewpoints;
this module decomposes each continuous hop into the closest
CityNav-action sequence so the trajectory log is submittable. Consumed
by the CityNav runner and ``flight_model.DiscreteFlightSimulator``.

The decomposition is greedy and deterministic: rotate toward bearing,
advance forward to the target XY, then step altitude to the target
height. Altitude is the last phase so the drone never descends until
it is over the target XY (the standard quadcopter waypoint pattern).
The achieved pose may differ from the requested end pose when the
discrete action set cannot reach it exactly or the sequence hits
``max_actions``; in that case the next hop re-plans from the achieved
pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class CityNavAction(str, Enum):
    """The six-element CityNav action set."""

    FORWARD = "move_forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    ASCEND = "ascend"
    DESCEND = "descend"
    STOP = "stop"


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


def _apply(pose: Pose, action: CityNavAction) -> Pose:
    """Advance a pose by one CityNav action."""
    if action is CityNavAction.STOP:
        return pose
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


def apply_action(pose: Pose, action: CityNavAction) -> Pose:
    """Public re-export of ``_apply`` so the benchmark loop can advance."""
    return _apply(pose, action)


def snap_to_actions(
    start: Pose,
    end: Pose,
    *,
    heading_tol_deg: float = 15.0,
    xy_tol_m: float = SNAP_XY_TOL_M,
    alt_tol_m: float = 1.0,
    max_actions: int = 1024,
) -> tuple[list[CityNavAction], Pose]:
    """Greedy decomposition of one compiler hop into CityNav actions.

    Returns ``(actions, achieved_pose)``. ``achieved_pose`` may differ
    from ``end`` when the discrete action set cannot exactly reach it.
    ``max_actions`` is a non-binding runaway guard; the per-episode
    budget is enforced one level up by the runner's ``_drive_to`` loop.
    """
    actions: list[CityNavAction] = []
    pose = start

    def _emit(action: CityNavAction) -> bool:
        nonlocal pose
        if len(actions) >= max_actions:
            return False
        actions.append(action)
        pose = _apply(pose, action)
        return True

    target_xy = (end.x, end.y)
    while len(actions) < max_actions:
        dx = target_xy[0] - pose.x
        dy = target_xy[1] - pose.y
        xy_err = math.hypot(dx, dy)
        alt_err = end.altitude_m - pose.altitude_m
        bearing = (
            bearing_deg((pose.x, pose.y), target_xy)
            if xy_err > xy_tol_m
            else pose.heading_deg
        )
        heading_err = _wrap_signed(bearing - pose.heading_deg)

        if xy_err <= xy_tol_m and abs(alt_err) <= alt_tol_m:
            break
        if xy_err > xy_tol_m and abs(heading_err) > heading_tol_deg:
            if not _emit(
                CityNavAction.TURN_LEFT if heading_err < 0 else CityNavAction.TURN_RIGHT
            ):
                break
            continue
        if xy_err > xy_tol_m:
            if not _emit(CityNavAction.FORWARD):
                break
            continue
        if abs(alt_err) > alt_tol_m:
            if not _emit(
                CityNavAction.ASCEND if alt_err > 0 else CityNavAction.DESCEND
            ):
                break
            continue
        break

    return actions, pose


__all__ = [
    "ALTITUDE_STEP_M",
    "FORWARD_STEP_M",
    "TURN_STEP_DEG",
    "CityNavAction",
    "Pose",
    "apply_action",
    "snap_to_actions",
]
