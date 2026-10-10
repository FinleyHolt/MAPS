"""CityNav Goal-Predictor scoring.

Given a predicted target `(x, y)`, the ground-truth target, the episode
start, and the agent's trajectory, returns CityNav's canonical metrics
(NE / SR / OSR / SPL).

NE measures WHERE THE AGENT ENDED UP, in metres from the goal. It is the
straight-line distance between the committed point and the true target,
and it carries no path term: an agent that flies a 3 km tour and commits
1 m from the goal scores NE 1. Readers regularly assume NE means
deviation from an optimal path, so every label that prints it says which
it is.

`spl_pos` is the metric that prices the route: Anderson SPL over the
flown xy path, the value the CityNav paper and its follow-ups report.
It rides ring rows only, beside the ring verdict it multiplies.

Coordinate convention:
  world XYZ with Z=up, metric
  grid[row, col]: row index varies with Y (north), col index varies with X (east)
  cell at (row, col) centers on world (x, y) where
    x = x_min + (col + 0.5) * cell_size_m
    y = y_min + (row + 0.5) * cell_size_m
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


def world_to_grid(
    xy: tuple[float, float],
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
) -> tuple[int, int]:
    """Snap a world (x, y) to (row, col), clipped to grid extent."""
    H, W = grid_shape
    col = int((xy[0] - bounds.x_min) / cell_size_m)
    row = int((xy[1] - bounds.y_min) / cell_size_m)
    row = max(0, min(row, H - 1))
    col = max(0, min(col, W - 1))
    return row, col


def path_length_m(path: list[tuple[float, float]] | None) -> float:
    """Sum of Euclidean distances along a pose sequence.

    Public because the offline efficiency reader
    (:mod:`benchmarks.citynav.analysis.flight_cost`) prices a stored run's
    flown path with it: one definition of "how far did it fly", so a reported
    path length and an SPL denominator can never drift apart.
    """
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
    """Anderson SPL over the executed flight path (the CityNav-comparable value).

    ``SPL = sr_pos * L_opt / max(L_opt, L_agent)`` with ``L_opt`` the
    straight-line distance from the first flown pose to the target (the
    shortest path in obstacle-free aerial space) and ``L_agent`` the length of
    the flown xy path. Requires a real trajectory (>=2 poses); returns ``None``
    otherwise so a degenerate single-pose record never fabricates SPL = SR.

    This is the single source of truth for SPL: the live native scorer calls it,
    and offline analysis backfills SPL through it for runs that predate the key.
    """
    if not pos_trajectory or len(pos_trajectory) < 2:
        return None
    pts = [tuple(p) for p in pos_trajectory]
    l_opt = float(np.hypot(pts[0][0] - target_xy[0], pts[0][1] - target_xy[1]))
    l_agent = path_length_m(pts)
    if l_opt > 0.0 and l_agent > 0.0:
        return round(float(sr_pos * l_opt / max(l_opt, l_agent)), 4)
    return float(sr_pos)


def score_goal_prediction_citynav_native(
    *,
    final_pos_xy: tuple[float, float],
    pred_xy: tuple[float, float],
    target_xy: tuple[float, float],
    pos_trajectory: list[tuple[float, float]] | None = None,
    pred_trajectory: list[tuple[float, float]] | None = None,
    success_dist_m: float = 20.0,
) -> dict:
    """Score one episode exactly as upstream ``gsamllavanav/evaluate.py``.

    Upstream ``eval_goal_predictor`` computes four metric families, all
    **2D** (``Point2D.dist_to`` over (x, y) only; the paper says
    "spherical" but the shipped code is strictly 2D, so a faithful
    reproduction matches the code):

      - ``*_pos_to_goal``: final agent pose vs target
      - ``*_pred_to_goal``: final predicted goal vs target
      - oracle = the minimum distance over the whole trajectory

    ``NE`` is the straight-line distance from the final pose to the goal
    and carries no route term: an agent that flies a 3 km tour and ends
    1 m from the goal scores NE 1.

    Upstream ships no SPL of its own, so the only SPL here is
    ``spl_pos``: Anderson over the flown path, the value the paper and
    its follow-ups report, and the one key that does price the route.
    Both upstream metric families are always emitted. The canonical
    ``NE``/``SR``/``OSR`` keys mirror ``pos_to_goal`` so
    :func:`aggregate` summarizes benchmark-comparable navigation scores.
    """

    def _d(a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    ne_pos = _d(final_pos_xy, target_xy)
    ne_pred = _d(pred_xy, target_xy)
    pos_traj = list(pos_trajectory) if pos_trajectory else [final_pos_xy]
    pred_traj = list(pred_trajectory) if pred_trajectory else [pred_xy]
    or_pos = min(_d(p, target_xy) for p in pos_traj)
    or_pred = min(_d(p, target_xy) for p in pred_traj)
    sr_pos = 1 if ne_pos <= success_dist_m else 0
    sr_pred = 1 if ne_pred <= success_dist_m else 0
    osr_pos = 1 if or_pos <= success_dist_m else 0
    osr_pred = 1 if or_pred <= success_dist_m else 0

    # SPL (Anderson) over the executed flight path -- the comparable value
    # the CityNav paper and follow-ups report, under an explicitly-named
    # key (see :func:`spl_pos_from_path`, the single source of truth
    # offline analysis also backfills through).
    spl_pos = spl_pos_from_path(target_xy, pos_traj, sr_pos)

    return {
        "ne_pos_to_goal": round(ne_pos, 2),
        "ne_pred_to_goal": round(ne_pred, 2),
        "sr_pos_to_goal": sr_pos,
        "sr_pred_to_goal": sr_pred,
        "osr_pos_to_goal": osr_pos,
        "osr_pred_to_goal": osr_pred,
        "min_dist_pos_to_target_m": round(or_pos, 2),
        # Anderson SPL over the flown path (comparable to the CityNav paper).
        "spl_pos": spl_pos,
        # Canonical mirrors (pos_to_goal family) so aggregate() summarizes
        # the CityNav navigation protocol, not submitted-target diagnostics.
        "NE": round(ne_pos, 2),
        "SR": sr_pos,
        "OSR": osr_pos,
        "scoring_basis": "final_pos_xy",
    }


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
    """Radius of the largest circle that fits inside ``poly``.

    Half the polygon's width at the widest point it can hold, so a long
    thin shape reads as narrow no matter how much area it covers — which
    is the sense in which a target is too small to aim at.
    """
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
    """Did the agent end up AT the target, given the target's real extent?

    This IS ``SR`` on a bench whose ``SUCCESS_CRITERION`` is ``"shape"`` --
    the verdict key it returns is the same one the ring scorer returns,
    because a row carries one success mask and one verdict under it.

    The target's shape substituted for a disc around its centre. Where an
    episode has annotated ground truth — a linked
    object's contour or a hand-drawn/derived mask — success is landing within
    ``tolerance_m`` of that shape, which at the default ``0.0`` means landing
    inside it. Where it has none, it falls back to a disc of ``radius_m``,
    exactly like ``SR`` but with the dataset's own radius.

    ``gt_polygon`` is the caller's choice of shape — the linked object's
    contour when the episode names one, else its ``target_mask``.

    ``floor_radius_m`` is the tightest target worth asking for: no shape
    scores stricter than a disc of that radius. A shape narrower than the
    floor is widened until a circle of ``floor_radius_m`` fits, rather than
    replaced by one — the footprint still says WHERE, the floor only says
    how finely the agent is asked to land. Road intersections here run
    4.1 m wide at the median while the agent's arrival on episodes it
    already gets right is 2.9 m, so without a floor most of them would be
    scored on a precision it has never demonstrated.
    """
    if gt_polygon and len(gt_polygon) >= 3:
        from shapely.geometry import Point

        poly = _as_polygon(gt_polygon)
        reach, floored = tolerance_m, False
        if floor_radius_m > 0.0:
            # _inradius_m is iterative and lands within a tolerance of the
            # true value, so ignore sub-millimetre gains — a shape that
            # already holds the floor circle must score exactly as drawn.
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


#: Every key on a row whose value is a success verdict or scales with one.
#: An episode that never flew has all of them zeroed together -- listing them
#: HERE rather than at each writer is the point: the zero-flight branch used
#: to zero the four ``*_to_goal`` flags and leave the mirrored ``SR``/``OSR``
#: alone, so an episode that never took off scored SR=1 whenever its start
#: happened to sit inside the mask.
_SUCCESS_FLAGS = (
    "SR",
    "OSR",
    "sr_pos_to_goal",
    "sr_pred_to_goal",
    "osr_pos_to_goal",
    "osr_pred_to_goal",
    # SPL multiplies the success bit, so a zeroed verdict zeroes it too.
    "spl_pos",
)

#: Ring-basis keys, dropped from a shape-scored row. Each answers "within R
#: of the target POINT", which is not the question that bench asks.
_RING_ONLY = (
    "sr_pos_to_goal",
    "sr_pred_to_goal",
    "OSR",
    "osr_pos_to_goal",
    "osr_pred_to_goal",
    "spl_pos",
)


def score_arrival(
    *,
    criterion: str,
    final_pos_xy: tuple[float, float],
    pred_xy: tuple[float, float],
    target_xy: tuple[float, float],
    pos_trajectory: list[tuple[float, float]] | None = None,
    pred_trajectory: list[tuple[float, float]] | None = None,
    success_dist_m: float = 20.0,
    gt_polygon: list[tuple[float, float]] | None = None,
    radius_m: float | None = None,
    tolerance_m: float = 0.0,
    floor_radius_m: float = 0.0,
    flew: bool = True,
) -> dict:
    """One episode's metrics, under THIS bench's success mask.

    ``criterion`` comes from ``paths.SUCCESS_CRITERION``, read at call time
    so the Cal City bridge's rebind reaches it:

      ``"ring"``   ``SR`` is 1 within ``success_dist_m`` of the target point
      ``"shape"``  ``SR`` is 1 inside the target's own annotated footprint

    ``SR`` is the verdict either way. There is no second success key and no
    per-row flag to reconcile: a reader that takes ``SR`` gets the number
    this bench is scored on. The criterion is stamped as
    ``success_criterion`` so a stored row says which mask produced it
    without anyone inferring it from coverage.

    Criterion-neutral keys ride every row: ``NE``, ``ne_pos_to_goal``,
    ``ne_pred_to_goal``, ``min_dist_pos_to_target_m`` -- raw metres to the
    target CENTROID, no radius in the computation. The shape criterion's
    analogous quantity is ``precise_distance_m``, which measures to the
    footprint's BOUNDARY. Both stay; they answer different questions.

    ``flew=False`` is the honest zero for an episode that never took off:
    every success flag goes to 0 while the distances stay true. Passing it
    is how a caller says "no flight", instead of naming the flags itself.

    Every writer of a scored row goes through here -- the main merge, the
    zero-flight branch, and the pre-flight provisional floor -- so the three
    cannot drift onto different criteria.
    """
    if criterion not in ("ring", "shape"):
        raise ValueError(f"unknown success criterion {criterion!r}")

    scores = score_goal_prediction_citynav_native(
        final_pos_xy=final_pos_xy,
        pred_xy=pred_xy,
        target_xy=target_xy,
        pos_trajectory=pos_trajectory,
        pred_trajectory=pred_trajectory,
        success_dist_m=success_dist_m,
    )
    if criterion == "shape":
        for key in _RING_ONLY:
            scores.pop(key, None)
        # Overwrites ``SR`` with the mask verdict -- the same key, because
        # it is the same question asked of this bench's own target.
        scores.update(
            score_precise(
                final_xy=pred_xy,
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
    keys_mean = [
        "NE",
        "SR",
        "OSR",
        "spl_pos",
        "ne_pos_to_goal",
        "ne_pred_to_goal",
        "sr_pos_to_goal",
        "sr_pred_to_goal",
        "osr_pos_to_goal",
        "osr_pred_to_goal",
        "min_dist_pos_to_target_m",
    ]
    out: dict = {"n_episodes": n}
    for k in keys_mean:
        # Skip absent keys and ``None`` values (``spl_pos`` on a degenerate
        # single-pose record) so a null never poisons the mean.
        vals = [r[k] for r in per_episode if r.get(k) is not None]
        if vals:
            out[k] = round(float(np.mean(vals)), 4)
    # A shape-scored SR pools two different tests -- rows with an annotated
    # footprint (containment) and rows without one (the dataset's disc) --
    # and how much of the corpus each covers moves as annotation lands.
    # Report them apart, so a change in the headline is attributable to real
    # ground truth rather than to the fallback standing in until a shape
    # exists. Absent on a ring bench, which resolves no ``precise_target``.
    for kind in ("shape", "radius"):
        vals = [
            r["SR"]
            for r in per_episode
            if r.get("precise_target") == kind and r.get("SR") is not None
        ]
        out[f"precise_{kind}_n"] = len(vals)
        if vals:
            out[f"SR_by_{kind}"] = round(float(np.mean(vals)), 4)
    # How many shapes were too narrow to score as drawn. A high count means
    # the headline is largely a floor-radius number wearing a shape's name.
    out["precise_floored_n"] = sum(
        1 for r in per_episode if r.get("precise_floor_applied")
    )
    # CityNav reports SR/OSR/SPL as percentages.
    for k in (
        "SR",
        "OSR",
        "spl_pos",
        "sr_pos_to_goal",
        "sr_pred_to_goal",
        "osr_pos_to_goal",
        "osr_pred_to_goal",
        "SR_by_shape",
        "SR_by_radius",
    ):
        if k in out:
            out[k + "_pct"] = round(out[k] * 100.0, 2)
    return out
