"""``union_gap``: the gap to a union's nearest member, pruned by member boxes
and seeded from the nearest box, equals the plain scan -- value AND witness."""
import numpy as np
import pytest

from maps.search.analyze_scene import geom_compute as G
from maps.search.analyze_scene.geom_program import union_gap
from maps.search.tests.fakes import ring


def _plain(members, item):
    """The scan with no bound at all -- every member, in order."""
    d = wr = wi = None
    for g in members:
        _d, _wr, _wi = G.gap_witness(g, item)
        if d is None or _d < d:
            d, wr, wi = _d, _wr, _wi
    return d, wr, wi


def _union_gap(members, item):
    return union_gap(members, G.stack_bboxes(members), item)


def _boxes(seed, n_members, n_items, span):
    rng = np.random.default_rng(seed)
    members = [G.mask_geom(ring(rng.uniform(0, span), rng.uniform(0, span),
                                 rng.uniform(4, 40), rng.uniform(4, 40)))
               for _ in range(n_members)]
    items = [G.mask_geom(ring(rng.uniform(0, span), rng.uniform(0, span),
                               rng.uniform(4, 40), rng.uniform(4, 40)))
             for _ in range(n_items)]
    return members, items


def test_the_scan_matches_the_plain_one_on_a_class_sized_reference(monkeypatch):
    members, items = _boxes(19, 400, 60, 3000)
    want = [_plain(members, item) for item in items]
    calls = [0]
    real = G.gap_witness

    def counting(a, b):
        calls[0] += 1
        return real(a, b)
    monkeypatch.setattr(G, "gap_witness", counting)
    for item, (d, wr, wi) in zip(items, want):
        got = _union_gap(members, item)
        assert got[0] == pytest.approx(d)
        assert got[1] == wr and got[2] == wi, "a different witness survived"
    # Vacuity floor: every assert above passes just as well if the bound
    # pruned nothing, so require it to have done real work.
    assert calls[0] < 0.25 * len(items) * len(members), calls[0]


def test_the_seed_does_not_move_a_tie():
    """Two members that tie exactly: the earlier one is still recorded, even
    when the seed evaluates the later one first."""
    item = G.mask_geom(ring(0, 0, 10, 10))
    left = G.mask_geom(ring(-30, 0, 10, 10))
    right = G.mask_geom(ring(30, 0, 10, 10))
    for members in ([left, right], [right, left]):
        want = _plain(members, item)
        got = _union_gap(members, item)
        assert got[0] == pytest.approx(want[0])
        assert got[1] == want[1] and got[2] == want[2]


def test_a_zero_gap_stops_the_scan_and_keeps_the_earliest_zero():
    a = G.mask_geom(ring(0, 0, 10, 10))
    overlapping = G.mask_geom(ring(5, 5, 10, 10))
    also = G.mask_geom(ring(6, 6, 10, 10))
    d, wr, _ = _union_gap([overlapping, also], a)
    first, fwr, _ = G.gap_witness(overlapping, a)
    assert d == 0.0 and first == 0.0 and wr == fwr


def test_a_bound_an_ulp_above_its_own_gap_still_evaluates_the_member():
    """The bound and the gap are reached by different routes and can disagree
    in the last bit."""
    members, items = _boxes(19, 400, 60, 3000)
    item = items[6]
    bounds = G.bbox_bounds(G.stack_bboxes(members), item)
    j = int(bounds.argmin())
    gap = G.gap_witness(members[j], item)[0]
    assert bounds[j] > gap, "the ulp inversion this guards is gone; re-derive"
    assert bounds[j] - gap < G.BOUND_SLACK_M
    got = _union_gap(members, item)
    assert got[0] is not None and got[0] == pytest.approx(_plain(members, item)[0])


def test_bbox_bounds_matches_bbox_gap_member_for_member():
    rng = np.random.default_rng(4)
    members = [G.mask_geom(ring(rng.uniform(0, 900), rng.uniform(0, 900),
                                 rng.uniform(2, 30), rng.uniform(2, 30)))
               for _ in range(120)]
    item = G.mask_geom(ring(400, 400, 20, 20))
    vec = G.bbox_bounds(G.stack_bboxes(members), item)
    one = np.asarray([G.bbox_gap(g, item) for g in members])
    assert np.allclose(vec, one)


def test_an_unknown_box_prunes_nothing():
    item = G.mask_geom(ring(0, 0, 10, 10))
    empty = G.mask_geom([])
    real = G.mask_geom(ring(500, 500, 10, 10))
    bounds = G.bbox_bounds(G.stack_bboxes([empty, real]), item)
    assert bounds[0] == 0.0, "an unknown box must not claim distance"
    assert bounds[1] > 100.0
    assert G.bbox_bounds(G.stack_bboxes([real]), empty).tolist() == [0.0]
    assert G.bbox_bounds(G.stack_bboxes([]), item).tolist() == []
