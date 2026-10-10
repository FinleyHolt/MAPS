"""``bbox_gap`` must be a TRUE lower bound, or the union min it prunes is wrong.

The union branch in ``geom_program.run_program`` takes the min gap over every
member of a class-sized reference. Skipping a member is only admissible if the
bound can never exceed the real gap: one counterexample and the reported
minimum is silently too large, on a table the model reads.

So the bound is checked against ``gap_witness`` itself over randomised pairs,
including the cases a box bound gets wrong if it forgets a case -- overlapping
boxes, containment, shared edges, and a point-mask degenerate.
"""
import math

import numpy as np
import pytest

from maps.search.analyze_scene import geom_compute as G


def _ring(x, y, w, h):
    return [[(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]]


def _pt(x, y):
    return [[(x, y)]]


def test_the_bound_never_exceeds_the_true_gap():
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(400):
        a = G.mask_geom(_ring(rng.uniform(-500, 500), rng.uniform(-500, 500),
                              rng.uniform(1, 80), rng.uniform(1, 80)))
        b = G.mask_geom(_ring(rng.uniform(-500, 500), rng.uniform(-500, 500),
                              rng.uniform(1, 80), rng.uniform(1, 80)))
        true_gap, _wa, _wb = G.gap_witness(a, b)
        bound = G.bbox_gap(a, b)
        assert bound <= true_gap + 1e-9, (bound, true_gap)
        checked += 1
    assert checked == 400


def test_the_bound_holds_for_containment_touch_and_points():
    outer = G.mask_geom(_ring(0, 0, 100, 100))
    inner = G.mask_geom(_ring(10, 10, 5, 5))
    touching = G.mask_geom(_ring(100, 0, 10, 10))
    far = G.mask_geom(_ring(1000, 1000, 10, 10))
    point_in = G.mask_geom(_pt(50, 50))
    for a, b in ((outer, inner), (outer, touching), (outer, far),
                 (outer, point_in), (inner, far), (point_in, far)):
        true_gap, _wa, _wb = G.gap_witness(a, b)
        assert G.bbox_gap(a, b) <= true_gap + 1e-9
    # and it is not vacuously zero: a far pair must actually prune
    assert G.bbox_gap(outer, far) > 100.0
    assert G.bbox_gap(outer, inner) == 0.0


def test_the_bound_is_symmetric_and_cached():
    a = G.mask_geom(_ring(0, 0, 10, 10))
    b = G.mask_geom(_ring(50, 50, 10, 10))
    assert G.bbox_gap(a, b) == pytest.approx(G.bbox_gap(b, a))
    assert G.geom_bbox(a) == (0.0, 0.0, 10.0, 10.0)
    assert a["BB"] == (0.0, 0.0, 10.0, 10.0)  # cached on the geom
    empty = G.mask_geom([])
    assert G.geom_bbox(empty) is None
    assert G.bbox_gap(a, empty) == 0.0  # prunes nothing rather than lying


def test_the_pruned_union_min_equals_the_unpruned_one():
    """The loop the bound serves, both ways, on a class-sized reference."""
    rng = np.random.default_rng(5)
    members = [G.mask_geom(_ring(rng.uniform(0, 2000), rng.uniform(0, 2000),
                                 rng.uniform(4, 40), rng.uniform(4, 40)))
               for _ in range(300)]
    items = [G.mask_geom(_ring(rng.uniform(0, 2000), rng.uniform(0, 2000),
                               rng.uniform(4, 40), rng.uniform(4, 40)))
             for _ in range(40)]
    saved = 0
    for item in items:
        plain = (None, None, None)
        for g in members:
            gd, gwr, gwi = G.gap_witness(g, item)
            if plain[0] is None or gd < plain[0]:
                plain = (gd, gwr, gwi)

        d = wr = wi = None
        calls = 0
        for g in members:
            if d is not None:
                if d <= 0.0:
                    break
                if G.bbox_gap(g, item) > d:
                    continue
            calls += 1
            _d, _wr, _wi = G.gap_witness(g, item)
            if d is None or _d < d:
                d, wr, wi = _d, _wr, _wi
        saved += len(members) - calls
        assert d == pytest.approx(plain[0]), "the pruned min differs"
        assert wr == plain[1] and wi == plain[2], "a different witness survived"
    # A vacuity floor: a run that pruned nothing would pass every assert above
    # while proving nothing about the bound doing any work.
    assert saved > len(items) * len(members) * 0.5, saved


def test_a_zero_gap_stops_the_scan_without_changing_the_answer():
    """Containment gives 0, and nothing can beat 0 -- but the break must not
    change WHICH witness is kept when an earlier member already tied."""
    a = G.mask_geom(_ring(0, 0, 10, 10))
    overlapping = G.mask_geom(_ring(5, 5, 10, 10))
    also = G.mask_geom(_ring(6, 6, 10, 10))
    members = [overlapping, also]
    d = wr = None
    for g in members:
        if d is not None and d <= 0.0:
            break
        _d, _wr, _wi = G.gap_witness(g, a)
        if d is None or _d < d:
            d, wr = _d, _wr
    assert d == 0.0
    first, fwr, _ = G.gap_witness(overlapping, a)
    assert first == 0.0 and wr == fwr, "the earliest zero must be the one kept"


def test_math_bound_matches_a_hand_computed_separation():
    a = G.mask_geom(_ring(0, 0, 10, 10))
    b = G.mask_geom(_ring(30, 40, 10, 10))
    assert G.bbox_gap(a, b) == pytest.approx(math.hypot(20.0, 30.0))
