"""Every box prune in the exact-mask kernels returns what the unpruned scan returns,
value and witness alike."""
import numpy as np
import pytest

from maps.query import spatial_math as G
from maps.mission.tests.fakes import ring


def _shapes(seed, n, span=600):
    """Rings, two-vertex ways and points, plus nested, touching and crossing pairs."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x, y = rng.uniform(-span / 2, span / 2, size=2)
        kind = rng.integers(0, 3)
        if kind == 0:
            out.append(ring(x, y, *rng.uniform(1, 60, size=2)))
        elif kind == 1:
            out.append([[(x, y), (x + rng.uniform(-80, 80), y + rng.uniform(-80, 80))]])
        else:
            out.append([[(x, y)]])
    out += [ring(0, 0, 100, 100), ring(10, 10, 5, 5), ring(100, 0, 10, 10),
            [[(-50, 50), (150, 50)]], ring(20, -20, 10, 300)]
    return [G.mask_geom(s) for s in out]


def _parity(P, A, B):
    """Even-odd membership over every edge, no prune."""
    Ax, Ay, Bx, By = A[:, 0][None], A[:, 1][None], B[:, 0][None], B[:, 1][None]
    px, py = P[:, 0][:, None], P[:, 1][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        xin = (Bx - Ax) * (py - Ay) / (By - Ay) + Ax
    return ((((Ay > py) != (By > py)) & (px < xin)).sum(axis=1) & 1).astype(bool)


def test_containment_is_the_unpruned_parity_across_chunks_holes_and_edges(monkeypatch):
    g = G.mask_geom(ring(0, 0, 100, 100) + ring(40, 40, 20, 20))
    rng = np.random.default_rng(3)
    edges = G.mask_geom([c for _ in range(200) for c in ring(
        *rng.uniform(0, 500, size=2), *rng.uniform(5, 30, size=2))])
    lattice = np.asarray([(x, y) for y in (0.0, 40.0, 50.0, 100.0, -1.0)
                          for x in (-1.0, 0.0, 40.0, 50.0, 100.0)], dtype=float)
    P = np.vstack([lattice, rng.uniform(0, 500, size=(400, 2))])
    monkeypatch.setattr(G, "_PAIR_CHUNK", len(edges["CA"]) * 4)   # many chunks
    for geom in (g, edges):
        want = _parity(P, geom["CA"], geom["CB"])
        assert np.array_equal(G._inside_mask_arr(P, geom["CA"], geom["CB"]), want)
    assert not G.mask_contains(g, [(50.0, 50.0)])[0], "the hole subtracts"


def test_the_pruned_point_to_segment_argmin_is_the_full_scan():
    rng = np.random.default_rng(7)
    pruned = 0
    for k in range(16):
        n_p, n_s = (rng.integers(50, 400), rng.integers(300, 2000)) if k % 2 else \
                   (rng.integers(2000, 6000), rng.integers(20, 120))
        P = rng.uniform(-500, 500, size=(int(n_p), 2))
        A = rng.uniform(-500, 500, size=(int(n_s), 2))
        B = A + rng.uniform(-40, 40, size=A.shape)
        assert G._min_pt_to_segs(P, A, B) == G._min_pt_to_segs_all(P, A, B)
        pruned += len(P) * len(A) > G._PRUNE_PAIRS
    assert 0 < pruned < 16, pruned            # both paths exercised


def test_the_box_gated_gap_and_crossing_are_the_ungated_ones(monkeypatch):
    geoms = _shapes(23, 60)
    pairs = [(a, b) for i, a in enumerate(geoms) for b in geoms[i:]]
    gated = [G.gap_witness(a, b) for a, b in pairs]
    crossed = [G._segments_cross(a["A"], a["B"], b["A"], b["B"]) for a, b in pairs]
    monkeypatch.setattr(G, "bbox_gap", lambda a, b: 0.0)
    monkeypatch.setattr(G, "_overlaps_box", lambda A, B, C, D: np.ones(len(A), bool))
    assert gated == [G.gap_witness(a, b) for a, b in pairs]
    assert crossed == [G._segments_cross(a["A"], a["B"], b["A"], b["B"]) for a, b in pairs]
    assert 0 < sum(c is not None for c in crossed) < len(pairs)


def test_the_box_bound_never_exceeds_the_gap_and_an_unknown_box_prunes_nothing():
    geoms = _shapes(11, 80, 1000)
    for a in geoms[:20]:
        for b in geoms:
            assert G.bbox_gap(a, b) <= G.gap_witness(a, b)[0] + 1e-9
        assert np.allclose(G.bbox_bounds(G.stack_bboxes(geoms), a),
                           [G.bbox_gap(g, a) for g in geoms])
    empty = G.mask_geom([])
    assert G.bbox_bounds(G.stack_bboxes([empty, geoms[0]]), geoms[1])[0] == 0.0
    assert G.bbox_gap(geoms[0], empty) == 0.0


def _plain(members, item):
    d = wr = wi = None
    for g in members:
        _d, _wr, _wi = G.gap_witness(g, item)
        if d is None or _d < d:
            d, wr, wi = _d, _wr, _wi
    return d, wr, wi


def _boxes(rng, n):
    return [G.mask_geom(ring(*rng.uniform(0, 3000, size=2), *rng.uniform(4, 40, size=2)))
            for _ in range(n)]


def test_set_gap_is_the_plain_member_scan_with_its_witness(monkeypatch):
    rng = np.random.default_rng(19)
    members, items = _boxes(rng, 400), _boxes(rng, 60)
    want = [_plain(members, item) for item in items]
    # One member's box bound sits an ulp ABOVE its own gap: the slack keeps it.
    bounds = G.bbox_bounds(G.stack_bboxes(members), items[6])
    j = int(bounds.argmin())
    assert 0 < bounds[j] - G.gap_witness(members[j], items[6])[0] < G.BOUND_SLACK_M
    calls = [0]
    real = G.gap_witness

    def counting(a, b):
        calls[0] += 1
        return real(a, b)
    monkeypatch.setattr(G, "gap_witness", counting)
    got = [G.set_gap(members, G.stack_bboxes(members), item) for item in items]
    assert got == want
    assert calls[0] < 0.25 * len(items) * len(members), calls[0]   # it pruned


@pytest.mark.parametrize("order", [0, 1])
def test_set_gap_keeps_the_earliest_member_on_a_tie_and_stops_at_contact(order):
    item = G.mask_geom(ring(0, 0, 10, 10))
    pair = [G.mask_geom(ring(-30, 0, 10, 10)), G.mask_geom(ring(30, 0, 10, 10))]
    pair = pair[::-1] if order else pair
    assert G.set_gap(pair, G.stack_bboxes(pair), item) == _plain(pair, item)
    touching = [G.mask_geom(ring(5, 5, 10, 10)), G.mask_geom(ring(6, 6, 10, 10))]
    assert G.set_gap(touching, G.stack_bboxes(touching), item) == \
        G.gap_witness(touching[0], item)
