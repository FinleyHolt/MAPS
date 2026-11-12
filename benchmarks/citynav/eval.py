"""CityNav scoring: NE / SR / OSR / SPL for one episode.

NE is the straight-line distance in metres from where the agent ended to the
goal, with no path term: a 3 km tour that commits 1 m from the goal scores
NE 1. ``spl_pos`` (Anderson SPL over the flown xy path) is the metric that
prices the route; it rides ring rows only, beside the verdict it multiplies.

World XYZ is metric with Z up; grid cell (row, col) centres on
x = x_min + (col + 0.5) * cell_size_m, y = y_min + (row + 0.5) * cell_size_m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from maps.query.spatial_math import point_in_polygon


@dataclass(frozen=True)
class WorldBounds:
    """Axis-aligned world extent for the tile, meters."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


def path_length_m(path: list[tuple[float, float]]) -> float:
    """Sum of Euclidean distances along a pose sequence: the SPL denominator."""
    pts = np.asarray(path, dtype=np.float64)
    deltas = np.diff(pts, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def spl_pos_from_path(
    target_xy: tuple[float, float],
    pos_trajectory: list[tuple[float, float]],
    sr_pos: int,
) -> float:
    """Anderson SPL over the flown path: ``sr_pos * L_opt / max(L_opt, L_agent)``,
    ``L_opt`` from the first flown pose to the target."""
    pts = [tuple(p) for p in pos_trajectory]
    l_opt = float(np.hypot(pts[0][0] - target_xy[0], pts[0][1] - target_xy[1]))
    l_agent = path_length_m(pts)
    if l_opt > 0.0 and l_agent > 0.0:
        return round(float(sr_pos * l_opt / max(l_opt, l_agent)), 4)
    return float(sr_pos)


def score_precise(
    *,
    final_xy: tuple[float, float],
    target_xy: tuple[float, float],
    gt_polygon: list[tuple[float, float]] | None,
    radius_m: float,
) -> dict:
    """``SR`` on a shape-criterion bench: did the agent end up inside the target's real extent?

    With a ``gt_polygon`` (the linked object's contour, else the episode's
    ``target_mask``), success is the point inside the shape as drawn, by the
    flight measures' even-odd test; without one, within a disc of ``radius_m``.
    """
    if gt_polygon and len(gt_polygon) >= 3:
        return {"SR": 1 if point_in_polygon(final_xy, gt_polygon) else 0, "precise_target": "shape"}
    distance = math.hypot(final_xy[0] - target_xy[0], final_xy[1] - target_xy[1])
    return {"SR": 1 if distance <= radius_m else 0, "precise_target": "radius"}


#: Every key that is a success verdict or scales with one (SPL multiplies it):
#: an episode that never flew has all of them zeroed together.
_SUCCESS_FLAGS = ("SR", "OSR", "spl_pos")

#: Ring-basis keys ("within R of the target point"), dropped from a shape-scored row.
_RING_ONLY = ("OSR", "spl_pos")


def score_arrival(
    *,
    criterion: str,
    final_pos_xy: tuple[float, float],
    target_xy: tuple[float, float],
    pos_trajectory: list[tuple[float, float]] | None = None,
    success_dist_m: float = 20.0,
    gt_polygon: list[tuple[float, float]] | None = None,
    commit_xy: tuple[float, float] | None = None,
    flew: bool = True,
) -> dict:
    """One episode's metrics under this bench's success mask; every scored row is written here.

    ``criterion`` (``Bench.success_criterion``): ``"ring"`` scores ``SR`` 1
    within ``success_dist_m`` of the target point, ``"shape"`` inside the
    target's annotated footprint at ``commit_xy`` (the commit point; the final
    pose when there is none). ``SR`` is the verdict either way, and the
    row is stamped ``success_criterion``. ``NE`` measures to the target
    centroid on every row.

    ``flew=False`` zeroes every success flag and keeps ``NE``.
    """
    if criterion not in ("ring", "shape"):
        raise ValueError(f"unknown success criterion {criterion!r}")

    # Scored in 2D as upstream ``gsamllavanav/evaluate.py`` does: NE from the
    # final pose, OSR from the trajectory's closest point. Upstream ships no SPL.
    def _d(p: tuple[float, float]) -> float:
        return float(np.hypot(p[0] - target_xy[0], p[1] - target_xy[1]))

    ne = _d(final_pos_xy)
    # Under two poses nothing was flown: the path is the final pose, held.
    pos_traj = list(pos_trajectory) if pos_trajectory and len(pos_trajectory) >= 2 else [final_pos_xy, final_pos_xy]
    sr = 1 if ne <= success_dist_m else 0
    scores = {
        "spl_pos": spl_pos_from_path(target_xy, pos_traj, sr),
        "NE": round(ne, 2),
        "SR": sr,
        "OSR": 1 if min(_d(p) for p in pos_traj) <= success_dist_m else 0,
    }
    if criterion == "shape":
        for key in _RING_ONLY:
            scores.pop(key, None)
        scores.update(
            score_precise(
                final_xy=final_pos_xy if commit_xy is None else commit_xy,
                target_xy=target_xy,
                gt_polygon=gt_polygon,
                radius_m=success_dist_m,
            )
        )
    scores["success_criterion"] = criterion
    if not flew:
        for key in _SUCCESS_FLAGS:
            if key in scores:
                scores[key] = 0
    return scores
