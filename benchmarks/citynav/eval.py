"""CityNav scoring: NE / SR / OSR / SPL for one episode, and their macro average.

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


@dataclass(frozen=True)
class WorldBounds:
    """Axis-aligned world extent for the tile, meters."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width_m(self) -> float:
        return self.x_max - self.x_min

    @property
    def height_m(self) -> float:
        return self.y_max - self.y_min


def path_length_m(path: list[tuple[float, float]] | None) -> float:
    """Sum of Euclidean distances along a pose sequence: the one definition
    of "how far did it fly" that path lengths and the SPL denominator share."""
    if not path or len(path) < 2:
        return 0.0
    pts = np.asarray(path, dtype=np.float64)
    deltas = np.diff(pts, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def spl_pos_from_path(
    target_xy: tuple[float, float],
    pos_trajectory: list[tuple[float, float]] | None,
    sr_pos: int,
) -> float | None:
    """Anderson SPL over the flown path: ``sr_pos * L_opt / max(L_opt, L_agent)``,
    ``L_opt`` from the first flown pose to the target; ``None`` under two poses."""
    if not pos_trajectory or len(pos_trajectory) < 2:
        return None
    pts = [tuple(p) for p in pos_trajectory]
    l_opt = float(np.hypot(pts[0][0] - target_xy[0], pts[0][1] - target_xy[1]))
    l_agent = path_length_m(pts)
    if l_opt > 0.0 and l_agent > 0.0:
        return round(float(sr_pos * l_opt / max(l_opt, l_agent)), 4)
    return float(sr_pos)


def _as_polygon(polygon: list[tuple[float, float]]):
    """Shapely polygon from a ring, self-touching vertices repaired."""
    from shapely.geometry import Polygon

    poly = Polygon(polygon).buffer(0)
    if poly.is_empty:
        raise ValueError("degenerate ground-truth polygon")
    return poly


# Reaches are reported to the millimetre; a gain under that is solver noise.
_FLOOR_EPS_M = 1e-3


def _inradius_m(poly) -> float:
    """Radius of the largest circle inside ``poly``: a long thin shape reads
    as narrow however much area it covers."""
    from shapely import maximum_inscribed_circle

    return float(maximum_inscribed_circle(poly).length)


def score_precise(
    *,
    final_xy: tuple[float, float],
    target_xy: tuple[float, float],
    gt_polygon: list[tuple[float, float]] | None,
    radius_m: float,
    tolerance_m: float = 0.0,
    floor_radius_m: float = 0.0,
) -> dict:
    """``SR`` on a shape-criterion bench: did the agent end up on the target's real extent?

    With a ``gt_polygon`` (the linked object's contour, else the episode's
    ``target_mask``), success is landing within ``tolerance_m`` of the shape
    (inside it at 0.0); without one, within a disc of ``radius_m``.

    ``floor_radius_m`` is the tightest target worth asking for: a shape
    narrower than a circle of that radius is widened until one fits.
    """
    if gt_polygon and len(gt_polygon) >= 3:
        from shapely.geometry import Point

        poly = _as_polygon(gt_polygon)
        reach, floored = tolerance_m, False
        if floor_radius_m > 0.0:
            # _inradius_m is iterative: a sub-millimetre gain is solver noise.
            widened = floor_radius_m - _inradius_m(poly)
            if widened > reach + _FLOOR_EPS_M:
                reach, floored = widened, True
        distance = float(Point(final_xy).distance(poly))
        return {
            "SR": 1 if distance <= reach else 0,
            "precise_target": "shape",
            "precise_distance_m": round(distance, 3),
            "precise_reach_m": round(reach, 3),
            "precise_floor_applied": floored,
        }
    distance = math.hypot(final_xy[0] - target_xy[0], final_xy[1] - target_xy[1])
    return {
        "SR": 1 if distance <= radius_m else 0,
        "precise_target": "radius",
        "precise_distance_m": round(distance, 3),
        "precise_reach_m": round(radius_m, 3),
        "precise_floor_applied": False,
    }


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
    radius_m: float | None = None,
    tolerance_m: float = 0.0,
    floor_radius_m: float = 0.0,
    flew: bool = True,
) -> dict:
    """One episode's metrics under this bench's success mask; every scored row is written here.

    ``criterion`` (``paths.SUCCESS_CRITERION``): ``"ring"`` scores ``SR`` 1
    within ``success_dist_m`` of the target point, ``"shape"`` inside the
    target's annotated footprint. ``SR`` is the verdict either way, and the
    row is stamped ``success_criterion``. ``NE`` measures to the target
    centroid on every row;
    ``precise_distance_m`` measures to the footprint boundary.

    ``flew=False`` zeroes every success flag and keeps the distances.
    """
    if criterion not in ("ring", "shape"):
        raise ValueError(f"unknown success criterion {criterion!r}")

    # Scored in 2D as upstream ``gsamllavanav/evaluate.py`` does: NE from the
    # final pose, OSR from the trajectory's closest point. Upstream ships no SPL.
    def _d(p: tuple[float, float]) -> float:
        return float(np.hypot(p[0] - target_xy[0], p[1] - target_xy[1]))

    ne = _d(final_pos_xy)
    pos_traj = list(pos_trajectory) if pos_trajectory else [final_pos_xy]
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
                final_xy=final_pos_xy,
                target_xy=target_xy,
                gt_polygon=gt_polygon,
                radius_m=success_dist_m if radius_m is None else radius_m,
                tolerance_m=tolerance_m,
                floor_radius_m=floor_radius_m,
            )
        )
    scores["success_criterion"] = criterion
    if not flew:
        for key in _SUCCESS_FLAGS:
            if key in scores:
                scores[key] = 0
    return scores


def aggregate(per_episode: list[dict]) -> dict:
    """Macro-average episode-level metrics into a single summary dict."""
    if not per_episode:
        return {"n_episodes": 0}
    n = len(per_episode)
    keys_mean = ["NE", "SR", "OSR", "spl_pos"]
    out: dict = {"n_episodes": n}
    for k in keys_mean:
        # A None (``spl_pos`` on a single-pose record) never poisons the mean.
        vals = [r[k] for r in per_episode if r.get(k) is not None]
        if vals:
            out[k] = round(float(np.mean(vals)), 4)
    # A shape-scored SR pools footprint containment and the disc fallback;
    # each is reported apart. A ring bench resolves no ``precise_target``.
    for kind in ("shape", "radius"):
        vals = [
            r["SR"]
            for r in per_episode
            if r.get("precise_target") == kind and r.get("SR") is not None
        ]
        out[f"precise_{kind}_n"] = len(vals)
        if vals:
            out[f"SR_by_{kind}"] = round(float(np.mean(vals)), 4)
    # Shapes too narrow to score as drawn: a high count makes SR a floor-radius number.
    out["precise_floored_n"] = sum(
        1 for r in per_episode if r.get("precise_floor_applied")
    )
    # CityNav reports SR/OSR/SPL as percentages.
    for k in ("SR", "OSR", "spl_pos", "SR_by_shape", "SR_by_radius"):
        if k in out:
            out[k + "_pct"] = round(out[k] * 100.0, 2)
    return out
