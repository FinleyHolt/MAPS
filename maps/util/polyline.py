"""Ring simplification and frame clipping for drawn footprints.

Two operations every layer that draws a mask polygon needs, and neither of
which may invent geometry: thin a closed ring toward a vertex budget without
cutting corners, and cut a ring to a rectangle without closing it along the
cut. Stdlib only.
"""

from __future__ import annotations

import math

__all__ = ["simplify_ring", "clip_ring_to_box"]


def _dp(pts: list, tol: float) -> list:
    """Douglas-Peucker on an OPEN run; both endpoints are pinned."""
    if len(pts) < 3:
        return list(pts)
    out = [pts[0]]
    # Iterative (an explicit stack, not recursion): a 2 m cell boundary loop
    # around a 418k-cell lake arrives with 1,566 vertices, and the recursive
    # form is O(n) deep on an already-thin run.
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
    out = [p for p, k in zip(pts, keep) if k]
    return out


def simplify_ring(loop, *, tolerance: float, max_points: int) -> list:
    """Thin a closed ring to at most ``max_points`` vertices, shape-aware.

    Douglas-Peucker at ``tolerance``, doubling the tolerance until the ring
    fits the budget. NEVER index-subsampling (``loop[::n]``), which is what
    this replaces: taking every n-th vertex of a raster boundary cuts every
    corner it lands between, and on a shape whose window shows only part of
    it the survivors land hundreds of metres apart and the ink reaches the
    reader as straight chords across the interior. Douglas-Peucker spends
    its budget where the curvature is, so a rectangle costs 4 vertices and a
    shoreline costs what it needs. Measured on Cal City's largest footprints
    (2 m cell quantization): a 1,566-vertex lake ring thins to 236 at one
    cell of tolerance, where the uniform rule kept every 16th vertex.

    ``tolerance`` is the deviation allowed in the ring's own units. Below
    the source's quantization it is lossless -- a 2 m cell boundary keeps
    its stair-steps, which are the honest realisation of what the raster
    knows. The doubling only bites on rings too detailed for the budget.
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


def _clip_segment(p0, p1, box):
    """Liang-Barsky: the part of segment p0->p1 inside ``box``, or None."""
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
    return ((x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy))


def clip_ring_to_box(loop, box) -> list:
    """The parts of a closed ring inside ``box``, as OPEN arcs.

    ``box`` is ``(x0, y0, x1, y1)``. Returns a list of point runs; a ring
    wholly inside comes back as one run with its first point repeated at
    the end, so a caller can draw every result the same open way.

    NOT polygon clipping. Sutherland-Hodgman and friends return a closed
    polygon, which means they synthesise edges ALONG the clip boundary --
    drawn, that says "the lake ends here" at the frame edge, which is a
    fact about the window masquerading as a fact about the world. Clipping
    edge-by-edge and leaving the crossings as gaps draws only geometry the
    ring actually has. Standard vector-tile practice is the same shape
    (clip with a buffer, never drop); the buffer belongs to the caller,
    which knows how wide its stroke is.
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
        seg = _clip_segment(pts[i], pts[(i + 1) % len(pts)], box)
        if seg is None:
            if len(run) >= 2:
                arcs.append(run)
            run = []
            continue
        a, b = seg
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
