"""ViSA's exploration path: the paper's Executor, step by step.

Contour merge ``C_merged = union of landmark contours``; a grid over its
bounding box with step ``delta = FOV * (1 - r)``, where ``FOV`` is the side
of the square the sensor surely covers at the flight altitude; a greedy set
cover of ``C_merged`` by FOV squares centred on grid points; a visitation
order that minimises the open path from the start.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from baselines.seam import Landmark

OVERLAP_R = 0.3
MAX_STOPS = 20


@dataclass
class CoveragePlan:
    """The plan and what it was built from, for the record and the figure."""

    waypoints: list[tuple[float, float]]
    fov_m: float
    grid_step_m: float
    grid_points: list[tuple[float, float]]
    contours: list[list[tuple[float, float]]]
    cells_total: int
    cells_covered: int


def merged_region(landmarks: list[Landmark], bounds):
    """The union of the matched landmarks' contours inside ``bounds`` (the
    search window); the window itself when none matched."""
    polys = []
    for lm in landmarks:
        try:
            p = Polygon(lm.contour).buffer(0)
        except (ValueError, TypeError):
            continue
        if not p.is_empty and p.area > 0:
            polys.append(p)
    window = box(bounds.x_min, bounds.y_min, bounds.x_max, bounds.y_max)
    if not polys:
        return window
    region = unary_union(polys).intersection(window)
    return region if not region.is_empty else window


def _axis_centres(lo: float, hi: float, fov: float, step: float) -> list[float]:
    extent = hi - lo
    if extent <= fov:
        return [(lo + hi) / 2.0]
    n = int(math.ceil((extent - fov) / step)) + 1
    return [lo + fov / 2.0 + i * step for i in range(n)]


def open_path_order(points: list[tuple[float, float]], start_xy) -> list[int]:
    """Order that minimises the open path start -> every point once.

    Exact over the permutations up to 8 points; nearest-neighbour seeded
    2-opt above that (the greedy cover caps the count at MAX_STOPS, so the
    heuristic never sees more than 20).
    """
    n = len(points)
    if n <= 1:
        return list(range(n))
    pts = np.asarray(points, dtype=float)
    s = np.asarray(start_xy, dtype=float)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    d0 = np.linalg.norm(pts - s, axis=-1)

    def cost(order):
        return d0[order[0]] + sum(d[order[i], order[i + 1]] for i in range(len(order) - 1))

    if n <= 8:
        return list(min(itertools.permutations(range(n)), key=cost))
    left = set(range(n))
    cur = int(np.argmin(d0))
    order = [cur]
    left.remove(cur)
    while left:
        cur = min(left, key=lambda j: d[cur, j])
        order.append(cur)
        left.remove(cur)
    improved = True
    while improved:
        improved = False
        for i in range(0, n - 1):
            for j in range(i + 2, n):
                cand = order[: i + 1] + order[i + 1 : j + 1][::-1] + order[j + 1 :]
                if cost(cand) + 1e-9 < cost(order):
                    order, improved = cand, True
    return order


def plan_coverage(landmarks: list[Landmark], bounds, start_xy: tuple[float, float], *, fov_m: float) -> CoveragePlan:
    region = merged_region(landmarks, bounds)
    fov = max(1.0, float(fov_m))
    step = fov * (1.0 - OVERLAP_R)
    minx, miny, maxx, maxy = region.bounds
    grid = [
        (x, y)
        for x in _axis_centres(minx, maxx, fov, step)
        for y in _axis_centres(miny, maxy, fov, step)
    ]
    # Rasterise the region: cell centres inside it are what a square must
    # cover. ponytail: a fixed fraction of the FOV, so a small landmark still
    # gets a few hundred cells and a whole block stays under ~10k.
    cell = max(1.0, fov / 25.0)
    cx = np.arange(minx + cell / 2.0, maxx, cell)
    cy = np.arange(miny + cell / 2.0, maxy, cell)
    if cx.size == 0:
        cx = np.array([(minx + maxx) / 2.0])
    if cy.size == 0:
        cy = np.array([(miny + maxy) / 2.0])
    gx, gy = np.meshgrid(cx, cy)
    gx, gy = gx.ravel(), gy.ravel()
    inside = shapely.contains_xy(region, gx, gy)
    if not inside.any():  # a sliver thinner than a cell: its centroid is the cell
        c = region.centroid
        gx, gy, inside = np.array([c.x]), np.array([c.y]), np.array([True])
    gx, gy = gx[inside], gy[inside]
    half = fov / 2.0
    covers = [
        (np.abs(gx - px) <= half) & (np.abs(gy - py) <= half) for (px, py) in grid
    ]
    covered = np.zeros(gx.size, dtype=bool)
    chosen: list[int] = []
    while len(chosen) < MAX_STOPS and not covered.all():
        gains = [int((c & ~covered).sum()) for c in covers]
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        chosen.append(best)
        covered |= covers[best]
    pts = [grid[i] for i in chosen]
    order = open_path_order(pts, start_xy)
    return CoveragePlan(
        waypoints=[pts[i] for i in order],
        fov_m=fov,
        grid_step_m=step,
        grid_points=grid,
        contours=[list(lm.contour) for lm in landmarks],
        cells_total=int(gx.size),
        cells_covered=int(covered.sum()),
    )
