"""``bbox_gap`` must be a TRUE lower bound, or the union min it prunes is wrong."""
import math

import numpy as np
import pytest

from maps.search.analyze_scene import geom_compute as G
from maps.search.tests.fakes import ring


def _pt(x, y):
    return [[(x, y)]]


def test_the_bound_never_exceeds_the_true_gap():
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(400):
        a = G.mask_geom(ring(rng.uniform(-500, 500), rng.uniform(-500, 500),
                              rng.uniform(1, 80), rng.uniform(1, 80)))
        b = G.mask_geom(ring(rng.uniform(-500, 500), rng.uniform(-500, 500),
                              rng.uniform(1, 80), rng.uniform(1, 80)))
        true_gap, _wa, _wb = G.gap_witness(a, b)
        bound = G.bbox_gap(a, b)
        assert bound <= true_gap + 1e-9, (bound, true_gap)
        checked += 1
    assert checked == 400


def test_the_bound_holds_for_containment_touch_and_points():
    outer = G.mask_geom(ring(0, 0, 100, 100))
    inner = G.mask_geom(ring(10, 10, 5, 5))
    touching = G.mask_geom(ring(100, 0, 10, 10))
    far = G.mask_geom(ring(1000, 1000, 10, 10))
    point_in = G.mask_geom(_pt(50, 50))
    for a, b in ((outer, inner), (outer, touching), (outer, far),
                 (outer, point_in), (inner, far), (point_in, far)):
        true_gap, _wa, _wb = G.gap_witness(a, b)
        assert G.bbox_gap(a, b) <= true_gap + 1e-9
    # and it is not vacuously zero: a far pair must actually prune
    assert G.bbox_gap(outer, far) > 100.0
    assert G.bbox_gap(outer, inner) == 0.0


def test_the_bound_is_symmetric_and_reads_the_vertex_box():
    a = G.mask_geom(ring(0, 0, 10, 10))
    b = G.mask_geom(ring(50, 50, 10, 10))
    assert G.bbox_gap(a, b) == pytest.approx(G.bbox_gap(b, a))
    assert a["BB"] == (0.0, 0.0, 10.0, 10.0)
    empty = G.mask_geom([])
    assert empty["BB"] is None
    assert G.bbox_gap(a, empty) == 0.0  # prunes nothing rather than lying


def test_math_bound_matches_a_hand_computed_separation():
    a = G.mask_geom(ring(0, 0, 10, 10))
    b = G.mask_geom(ring(30, 40, 10, 10))
    assert G.bbox_gap(a, b) == pytest.approx(math.hypot(20.0, 30.0))
