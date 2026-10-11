"""Ring simplification and frame clipping for drawn footprints, neither of
which invents geometry. Stdlib only."""

from __future__ import annotations

import math

__all__ = ["simplify_ring", "clip_ring_to_box"]


def _dp(pts: list, tol: float) -> list:
    """Douglas-Peucker on an OPEN run; both endpoints are pinned."""
    if len(pts) < 3:
        return list(pts)
    # An explicit stack: recursion is O(n) deep on an already-thin run.
    stack = [(0, len(pts) - 1)]
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        x0, y0 = pts[i0]
        x1, y1 = pts[i1]
        dx, dy = x1 - x0, y1 - y0
        span = math.hypot(dx, dy)
        worst_i, worst_d = i0, -1.0
        for i in range(i0 + 1, i1):
            px, py = pts[i]
            d = (math.hypot(px - x0, py - y0) if span == 0.0
                 else abs(dy * px - dx * py + x1 * y0 - y1 * x0) / span)
            if d > worst_d:
                worst_i, worst_d = i, d
        if worst_d > tol:
            keep[worst_i] = True
            stack.append((i0, worst_i))
            stack.append((worst_i, i1))
    return [p for p, k in zip(pts, keep) if k]


def simplify_ring(loop, *, tolerance: float, max_points: int) -> list:
    """Thin a closed ring to at most ``max_points`` vertices: Douglas-Peucker
    at ``tolerance`` (in the ring's units), doubling it until the ring fits.
    The budget goes where the curvature is; index-subsampling would cut
    corners into chords across the interior.
    """
    pts = [(float(p[0]), float(p[1])) for p in loop
           if isinstance(p, (list, tuple)) and len(p) == 2]
    if len(pts) < 3:
        return [[p[0], p[1]] for p in pts]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    tol = max(float(tolerance), 1e-9)
    thin = _dp(pts, tol)
    # The seam is pinned by construction (DP keeps both endpoints), so the
    # ring can only ever be over budget, never mis-closed.
    while len(thin) - 1 > int(max_points) and tol < 1e6:
        tol *= 2.0
        thin = _dp(pts, tol)
    return [[p[0], p[1]] for p in thin[:-1]]


def clip_t(p0, p1, box):
    """Liang-Barsky: the ``(t_enter, t_exit)`` of segment p0->p1 inside
    ``box`` = ``(x0, y0, x1, y1)``, or None."""
    x0, y0 = p0
    x1, y1 = p1
    bx0, by0, bx1, by1 = box
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - bx0), (dx, bx1 - x0),
                 (-dy, y0 - by0), (dy, by1 - y0)):
        if p == 0.0:
            if q < 0.0:
                return None          # parallel to this edge and outside it
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    if t0 > t1:
        return None
    return t0, t1


def clip_ring_to_box(loop, box) -> list:
    """The parts of a closed ring inside ``box`` = ``(x0, y0, x1, y1)``, as
    open point runs (a ring wholly inside: one run repeating its first point).
    Edge-by-edge, so no edge is synthesised along the box as polygon clipping
    would; a stroke buffer is the caller's.
    """
    pts = [(float(p[0]), float(p[1])) for p in loop
           if isinstance(p, (list, tuple)) and len(p) == 2]
    if len(pts) < 3:
        return []
    bx0, by0, bx1, by1 = (float(v) for v in box)
    box = (min(bx0, bx1), min(by0, by1), max(bx0, bx1), max(by0, by1))
    arcs: list = []
    run: list = []
    for i in range(len(pts)):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % len(pts)]
        t = clip_t((x0, y0), (x1, y1), box)
        if t is None:
            if len(run) >= 2:
                arcs.append(run)
            run = []
            continue
        dx, dy = x1 - x0, y1 - y0
        a, b = (x0 + t[0] * dx, y0 + t[0] * dy), (x0 + t[1] * dx, y0 + t[1] * dy)
        if run and run[-1] == a:
            run.append(b)
        else:
            if len(run) >= 2:
                arcs.append(run)
            run = [a, b]
    if len(run) >= 2:
        arcs.append(run)
    # A ring whose seam fell inside the box opens at the seam and closes
    # again at the end of the walk; rejoin those two so the drawn arc has no
    # phantom break at an arbitrary vertex.
    if len(arcs) > 1 and arcs[0][0] == arcs[-1][-1]:
        arcs[0] = arcs[-1] + arcs[0][1:]
        arcs.pop()
    return [[list(p) for p in a] for a in arcs]
