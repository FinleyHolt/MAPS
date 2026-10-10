"""Ring simplification keeps the shape; frame clipping never closes the cut."""

from __future__ import annotations

import math

from maps.util.polyline import clip_ring_to_box, simplify_ring

# A 2 m-quantized staircase ring around a 40 x 20 m block: the shape every
# prior footprint arrives as (cells_boundary_loops output).
def _staircase(w_m: float, h_m: float, cell: float = 2.0) -> list:
    ring = []
    x = 0.0
    while x < w_m:                       # south edge, stepping east
        ring += [[x, 0.0], [x + cell, 0.0]]
        x += cell
    y = 0.0
    while y < h_m:                       # east edge, stepping north
        ring += [[w_m, y], [w_m, y + cell]]
        y += cell
    x = w_m
    while x > 0.0:
        ring += [[x, h_m], [x - cell, h_m]]
        x -= cell
    y = h_m
    while y > 0.0:
        ring += [[0.0, y], [0.0, y - cell]]
        y -= cell
    return ring


def test_a_rectangle_costs_four_vertices_not_ninety_six():
    ring = _staircase(200.0, 100.0)
    assert len(ring) > 96, len(ring)          # vacuity: the input is dense
    out = simplify_ring(ring, tolerance=0.5, max_points=256)
    # Collinear runs collapse; only the corners survive. The old uniform
    # rule kept 96 vertices of this, every one of them redundant.
    assert len(out) <= 8, out
    xs = [p[0] for p in out]
    ys = [p[1] for p in out]
    assert (min(xs), max(xs), min(ys), max(ys)) == (0.0, 200.0, 0.0, 100.0)


def test_the_budget_tightens_the_tolerance_and_never_subsamples():
    # A circle at 1 m sampling: 600 vertices, no collinear runs to collapse,
    # so the budget can only be met by raising the tolerance.
    ring = [[300.0 * math.cos(t / 600.0 * 2 * math.pi),
             300.0 * math.sin(t / 600.0 * 2 * math.pi)] for t in range(600)]
    out = simplify_ring(ring, tolerance=0.05, max_points=64)
    assert len(out) <= 64, len(out)
    # Every survivor is a REAL vertex of the input, and the ring still
    # rounds: max radial error stays small, which index-subsampling
    # (which would also fit the budget) cannot promise on a shape whose
    # vertices are not uniformly informative.
    radii = [math.hypot(*p) for p in out]
    assert max(abs(r - 300.0) for r in radii) < 1e-6, max(radii)
    assert len(out) >= 8, out                 # vacuity: it did not collapse


def test_a_ring_inside_the_box_survives_whole_and_closed():
    ring = [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]]
    arcs = clip_ring_to_box(ring, (0.0, 0.0, 100.0, 100.0))
    assert len(arcs) == 1, arcs
    # first point repeated at the end -> drawing it OPEN reproduces the ring
    assert arcs[0][0] == arcs[0][-1], arcs[0]
    assert len(arcs[0]) == 5, arcs[0]


def test_a_ring_larger_than_the_box_draws_arcs_and_no_edge_along_the_cut():
    # The lake case: the shape runs well past the window on every side.
    ring = [[-500.0, -500.0], [500.0, -500.0], [500.0, 500.0], [-500.0, 500.0]]
    box = (0.0, 0.0, 100.0, 100.0)
    arcs = clip_ring_to_box(ring, box)
    # Every drawn point is on the box boundary or inside it...
    for a in arcs:
        for x, y in a:
            assert -1e-9 <= x <= 100.0 + 1e-9 and -1e-9 <= y <= 100.0 + 1e-9
    # ...and NOTHING is drawn: all four edges pass outside the box, so
    # there is no in-frame geometry. The old rule dropped the shape here
    # too; the difference shows in the partial case below.
    assert arcs == [], arcs


def test_a_partly_visible_ring_draws_its_in_frame_arc_only():
    # A shape crossing the right edge. The two crossing edges clip to
    # stubs; the far side is gone. Critically, no segment runs ALONG x=100
    # joining the stubs -- that would tell the reader the shape ends there.
    ring = [[50.0, 40.0], [400.0, 40.0], [400.0, 60.0], [50.0, 60.0]]
    arcs = clip_ring_to_box(ring, (0.0, 0.0, 100.0, 100.0))
    assert arcs, arcs
    drawn = [tuple(p) for a in arcs for p in a]
    assert any(abs(x - 100.0) < 1e-9 for x, _ in drawn), drawn   # it reaches
    # no drawn EDGE lies along the cut
    for a in arcs:
        for (x0, y0), (x1, y1) in zip(a, a[1:]):
            on_cut = abs(x0 - 100.0) < 1e-9 and abs(x1 - 100.0) < 1e-9
            assert not on_cut, (a, "edge synthesised along the clip boundary")


def test_the_seam_is_not_a_phantom_break():
    # The ring's first vertex sits inside the box, so the walk opens an arc
    # at the seam and closes another at the end. They are one arc.
    ring = [[50.0, 50.0], [50.0, 200.0], [60.0, 200.0], [60.0, 50.0]]
    arcs = clip_ring_to_box(ring, (0.0, 0.0, 100.0, 100.0))
    assert len(arcs) == 1, arcs
