"""Deterministic replay of the flown stops + the committed final leg in the
CityNav six-action alphabet (honest truncation; the final pose is what
scores).
"""

from __future__ import annotations

import logging
import math

from benchmarks.citynav.action_snap import (
    ARRIVAL_BOUND_M,
    FORWARD_STEP_M,
    SNAP_XY_TOL_M,
    Pose,
    apply_action,
    snap_to_actions,
)
from benchmarks.run_record.config import RunConfig

log = logging.getLogger("run_benchmark")


def synthesize_compliant_trajectory(
    *,
    cfg: RunConfig,
    flights: list[dict],
    start_pose_xyh: tuple[float, float, float],
    start_alt: float,
) -> tuple[list[tuple[float, float]], int]:
    """Replay the mission's flights (``MissionSession.flights``, at least one)
    through the six-action snap under the ``cfg.max_steps`` budget; the commit
    is the last flight's point, flown at cruise altitude with a reserve of
    ``min(6, max_steps // 2)`` actions. When the commit leg still falls short,
    the replay runs again with a larger reserve, dropping more of the stop tail,
    until the commit is reached or the whole budget is reserved. Returns the
    submittable trajectory (the xy after each action, the last the final pose;
    none when no action was needed) and how many flown stops before the commit
    it dropped.
    """
    tour_wps = [(float(f["resolved_xy"][0]), float(f["resolved_xy"][1])) for f in flights]
    tour_alts = [float(f["altitude_m"]) for f in flights]
    tour_fids = [str(f["target"]) for f in flights]
    descent_xy = tour_wps[-1]

    max_frames = int(cfg.max_steps)
    pose = Pose(0.0, 0.0, 0.0, 0.0)  # set from the start pose per attempt
    visited: list[tuple[float, float]] = []
    reached = 0  # stops before the commit the current attempt flew to

    def _drive_to(target_xy: tuple[float, float], *, altitude_m: float, budget: int) -> bool:
        """Fly one leg, re-aiming from the current pose until within
        ``SNAP_XY_TOL_M``. True when the budget TRUNCATED it farther than
        ``ARRIVAL_BOUND_M`` (the lattice's closest approach) from its target."""
        nonlocal pose
        limit = min(int(budget), max_frames)
        while len(visited) < limit:
            target = Pose(float(target_xy[0]), float(target_xy[1]), altitude_m, pose.heading_deg)
            actions, _ = snap_to_actions(pose, target)
            if not actions:
                break  # already at target
            for action in actions[: limit - len(visited)]:
                pose = apply_action(pose, action)
                visited.append((pose.x, pose.y))
            dx = pose.x - float(target_xy[0])
            dy = pose.y - float(target_xy[1])
            if math.hypot(dx, dy) <= SNAP_XY_TOL_M:
                break
        dx = pose.x - float(target_xy[0])
        dy = pose.y - float(target_xy[1])
        return len(visited) >= limit and math.hypot(dx, dy) > ARRIVAL_BOUND_M

    def _walk_tour(tour_budget: int, descent_reserve: int) -> bool:
        """Fly each stop in turn, every leg capped at ``tour_budget``. True
        when the reserve dropped the stop tail."""
        nonlocal reached
        reached = 0
        for i, wp in enumerate(tour_wps):
            if len(visited) >= tour_budget:
                log.warning(
                    "replay budget: the flown-stop walk hit the commit reserve at "
                    "step %d/%d (reserved %d) after %d/%d stops; remaining "
                    "tail stops dropped from the replay so the commit leg "
                    "is reached",
                    len(visited),
                    max_frames,
                    descent_reserve,
                    i,
                    len(tour_wps),
                )
                return True
            if _drive_to(wp, altitude_m=tour_alts[i], budget=tour_budget):
                log.warning(
                    "replay budget: leg %d/%d truncated MID-LEG at the commit "
                    "reserve (step %d/%d, reserved %d); remaining stops "
                    "dropped from the replay so the commit leg is reached",
                    i,
                    len(tour_wps),
                    len(visited),
                    max_frames,
                    descent_reserve,
                )
                return True
            reached = i + 1
        return False

    def _fly_descent() -> bool:
        """The final transit to the commit, aimed one snap tolerance past it
        so the approach does not stop short. True when the budget truncated it."""
        if len(visited) >= max_frames:
            return True
        final_target = descent_xy
        dx = descent_xy[0] - pose.x
        dy = descent_xy[1] - pose.y
        dist = math.hypot(dx, dy)
        if dist > SNAP_XY_TOL_M + 0.5:
            final_target = (
                descent_xy[0] + SNAP_XY_TOL_M * dx / dist,
                descent_xy[1] + SNAP_XY_TOL_M * dy / dist,
            )
        return _drive_to(final_target, altitude_m=pose.altitude_m, budget=max_frames)

    descent_reserve = min(6, max_frames // 2)
    descent_replans = 0
    while True:
        # A fresh replay from the start pose per attempt.
        pose = Pose(
            float(start_pose_xyh[0]),
            float(start_pose_xyh[1]),
            float(start_alt),
            float(start_pose_xyh[2]) % 360.0,
        )
        visited.clear()
        tail_dropped = _walk_tour(max_frames - descent_reserve, descent_reserve)
        descent_truncated = _fly_descent()
        if not descent_truncated:
            break
        short_m = math.hypot(pose.x - descent_xy[0], pose.y - descent_xy[1])
        if descent_reserve >= max_frames:
            log.error(
                "replay: commit descent UNREACHABLE within the "
                "action budget even with the FULL budget reserved (whole "
                "tour dropped): the final pose stays %.1fm short of the committed "
                "target %s (steps=%d/%d, replans=%d)",
                short_m,
                descent_xy,
                len(visited),
                int(max_frames),
                descent_replans,
            )
            break
        descent_replans += 1
        new_reserve = min(
            max_frames,
            max(2 * descent_reserve, descent_reserve + int(short_m / FORWARD_STEP_M) + 12, 16),
        )
        log.warning(
            "replay: commit descent truncated %.1fm short of "
            "the committed target %s (steps=%d/%d, reserve=%d) -> "
            "re-synthesizing the replay with reserve=%d (replan %d; the "
            "stop tail is already perceived, the commit transit is not "
            "optional)",
            short_m,
            descent_xy,
            len(visited),
            int(max_frames),
            descent_reserve,
            new_reserve,
            descent_replans,
        )
        descent_reserve = new_reserve

    final_pos_xy = (pose.x, pose.y)
    log.info(
        "replay trajectory (discrete 6-action): %d tour stops %s "
        "descent=%s frames=%d final_pos_xy=%s (budget=%d) "
        "tail_dropped=%s descent_truncated=%s",
        len(tour_fids),
        tour_fids,
        descent_xy,
        len(visited),
        final_pos_xy,
        int(max_frames),
        tail_dropped,
        descent_truncated,
    )
    # The commit is the tour's last stop and the descent flies it: only the stops before it drop.
    return visited, len(tour_wps) - 1 - min(reached, len(tour_wps) - 1)
