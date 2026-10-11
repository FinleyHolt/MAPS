"""Seeding the union scan from the nearest box must not move the answer.

The seed evaluates one member out of order to get an upper bound early. That
is only admissible if it changes which members are SKIPPED and never which one
is RECORDED: the minimum, and the witness pair a tie keeps, both have to match
the plain ordered scan exactly.

The oracle here is that plain scan, computed in the test, so neither side can
drift into agreement with a stored answer.
"""
import numpy as np
import pytest

from maps.search.analyze_scene import geom_compute as G
from maps.search.tests.fakes import ring


def _plain(members, item):
    """The scan with no bound at all -- every member, in order."""
    d = wr = wi = None
    for g in members:
        _d, _wr, _wi = G.gap_witness(g, item)
        if d is None or _d < d:
            d, wr, wi = _d, _wr, _wi
    return d, wr, wi


def _seeded(members, item):
    """The shipped scan: stacked bounds, nearest-box seed, ordered record."""
    bbs = G.stack_bboxes(members)
    bounds = G.bbox_bounds(bbs, item)
    ub = None
    seed_i, seed = -1, None
    if len(members) > 1 and len(bounds):
        seed_i = int(bounds.argmin())
        seed = G.gap_witness(members[seed_i], item)
        ub = seed[0]
    d = wr = wi = None
    calls = 0
    for gi, g in enumerate(members):
        if d is not None and d <= 0.0:
            break
        cut = d if d is not None else ub
        if cut is not None and bounds[gi] > cut + G.BOUND_SLACK_M:
            continue
        calls += 1
        # the seed member's triple is reused, never re-measured
        _d, _wr, _wi = seed if gi == seed_i else G.gap_witness(g, item)
        if d is None or _d < d:
            d, wr, wi = _d, _wr, _wi
    return (d, wr, wi), calls


def test_the_seeded_scan_matches_the_plain_one_on_a_class_sized_reference():
    rng = np.random.default_rng(19)
    members = [G.mask_geom(ring(rng.uniform(0, 3000), rng.uniform(0, 3000),
                                 rng.uniform(4, 40), rng.uniform(4, 40)))
               for _ in range(400)]
    items = [G.mask_geom(ring(rng.uniform(0, 3000), rng.uniform(0, 3000),
                               rng.uniform(4, 40), rng.uniform(4, 40)))
             for _ in range(60)]
    total_calls = 0
    for item in items:
        want = _plain(members, item)
        (got_d, got_wr, got_wi), calls = _seeded(members, item)
        total_calls += calls
        assert got_d == pytest.approx(want[0])
        assert got_wr == want[1], "a different witness survived the seed"
        assert got_wi == want[2], "a different witness survived the seed"
    # Vacuity floor: every assert above passes just as well if the seed pruned
    # nothing, so require it to have done real work.
    assert total_calls < 0.25 * len(items) * len(members), total_calls


def test_the_seed_does_not_move_a_tie():
    """Two members that tie exactly: the earlier one must still be recorded,
    even when the seed evaluates the later one first."""
    item = G.mask_geom(ring(0, 0, 10, 10))
    left = G.mask_geom(ring(-30, 0, 10, 10))
    right = G.mask_geom(ring(30, 0, 10, 10))
    for members in ([left, right], [right, left]):
        want = _plain(members, item)
        got, _calls = _seeded(members, item)
        assert got[0] == pytest.approx(want[0])
        assert got[1] == want[1] and got[2] == want[2]


def test_a_bound_an_ulp_above_its_own_gap_still_evaluates_the_member():
    """The bound and the gap are reached by different routes and can disagree
    in the last bit. Without slack the member holding the true minimum is
    skipped and the scan returns nothing at all -- the exact pair that failed
    is rebuilt here so a future tightening of BOUND_SLACK_M fails loudly."""
    rng = np.random.default_rng(19)
    members = [G.mask_geom(ring(rng.uniform(0, 3000), rng.uniform(0, 3000),
                                 rng.uniform(4, 40), rng.uniform(4, 40)))
               for _ in range(400)]
    items = [G.mask_geom(ring(rng.uniform(0, 3000), rng.uniform(0, 3000),
                               rng.uniform(4, 40), rng.uniform(4, 40)))
             for _ in range(60)]
    item = items[6]
    bounds = G.bbox_bounds(G.stack_bboxes(members), item)
    j = int(bounds.argmin())
    gap = G.gap_witness(members[j], item)[0]
    assert bounds[j] > gap, "the ulp inversion this guards is gone; re-derive"
    assert bounds[j] - gap < G.BOUND_SLACK_M
    got, _calls = _seeded(members, item)
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
