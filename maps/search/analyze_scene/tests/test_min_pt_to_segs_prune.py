"""The pruned point-to-segment argmin is the full scan's, pair for pair.

The oracle is the full scan itself (``_min_pt_to_segs_all``), so the
pruned path cannot drift into agreeing with a stored answer. Sets are
built so both prunings bite -- many points against few segments and few
points against many -- and so ties exist, where the witness pair is the
part a lower-bound cut could get wrong.
"""
import numpy as np

from maps.search.analyze_scene import geom_compute as G


def _segs(rng, n, lo=-500, hi=500, span=40):
    A = rng.uniform(lo, hi, size=(n, 2))
    B = A + rng.uniform(-span, span, size=(n, 2))
    return A, B


def _grid_pts(n):
    # a lattice: many exact ties in distance to axis-aligned segments
    xs = np.arange(n, dtype=float)
    return np.array([[x, y] for x in xs for y in xs])


def test_pruned_argmin_is_the_full_scan_on_both_shapes():
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(12):
        cases.append((rng.uniform(-500, 500, size=(int(rng.integers(50, 400)), 2)),
                      *_segs(rng, int(rng.integers(300, 2000)))))
        cases.append((rng.uniform(-500, 500, size=(int(rng.integers(2000, 6000)), 2)),
                      *_segs(rng, int(rng.integers(20, 120)))))
    # ties: lattice points against axis-aligned segments
    cases.append((_grid_pts(60), np.array([[10.0, -5.0], [10.0, 100.0], [-3.0, 20.0]]),
                  np.array([[10.0, 70.0], [50.0, 100.0], [70.0, 20.0]])))
    cases.append((_grid_pts(400)[:120000:7],
                  np.array([[100.0, 100.0]]), np.array([[100.0, 100.0]])))
    pruned = 0
    for P, A, B in cases:
        got = G._min_pt_to_segs(P, A, B)
        want = G._min_pt_to_segs_all(P, A, B)
        assert got == want, (got, want, P.shape, A.shape)
        pruned += len(P) * len(A) > G._PRUNE_PAIRS
    assert 0 < pruned < len(cases), pruned      # both paths exercised


def test_gap_witness_over_a_union_as_one_geom_is_unchanged():
    rng = np.random.default_rng(9)
    rings = []
    for _ in range(600):
        x, y = rng.uniform(0, 3000, size=2)
        w, h = rng.uniform(4, 30, size=2)
        rings.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
    union = G.mask_geom(rings)
    assert len(union["A"]) * 121 > G._PRUNE_PAIRS
    for _ in range(20):
        cx, cy = rng.uniform(0, 3000, size=2)
        ang = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        car = [(cx + 3 * np.cos(a), cy + 2 * np.sin(a)) for a in ang]
        car.append(car[0])
        item = G.mask_geom([car])
        d, wa, wb = G.gap_witness(union, item)
        # the oracle: the full scan both ways, the way gap_witness combines them
        best = (float("inf"), None, None)
        dd, i, foot = G._min_pt_to_segs_all(union["V"], item["A"], item["B"])
        if dd < best[0]:
            best = (dd, (float(union["V"][i, 0]), float(union["V"][i, 1])), foot)
        dd, i, foot = G._min_pt_to_segs_all(item["V"], union["A"], union["B"])
        if dd < best[0]:
            best = (dd, foot, (float(item["V"][i, 0]), float(item["V"][i, 1])))
        if d > 0.0:
            assert (d, wa, wb) == best, ((d, wa, wb), best)
