"""The gap's containment and crossing tests are gated on box overlap, exactly."""
import math

import numpy as np

from maps.search.analyze_scene import geom_compute as G
from maps.search.tests.fakes import ring


def _way(x, y, dx, dy):
    return [[(x, y), (x + dx, y + dy)]]


def _ungated(ga, gb):
    """gap_witness with no box gate: the three tests on every pair."""
    Va, Vb = ga["V"], gb["V"]
    if len(gb["CA"]):
        ins = G._inside_mask_arr(Va, gb["CA"], gb["CB"])
        if ins.any():
            k = int(np.argmax(ins))
            p = (float(Va[k, 0]), float(Va[k, 1]))
            return 0.0, p, p
    if len(ga["CA"]):
        ins = G._inside_mask_arr(Vb, ga["CA"], ga["CB"])
        if ins.any():
            k = int(np.argmax(ins))
            p = (float(Vb[k, 0]), float(Vb[k, 1]))
            return 0.0, p, p
    x = G._segments_cross(ga["A"], ga["B"], gb["A"], gb["B"])
    if x is not None:
        return 0.0, x, x
    best = (float("inf"), None, None)
    d, i, foot = G._min_pt_to_segs(Va, gb["A"], gb["B"])
    if d < best[0]:
        best = (d, (float(Va[i, 0]), float(Va[i, 1])), foot)
    d, i, foot = G._min_pt_to_segs(Vb, ga["A"], ga["B"])
    if d < best[0]:
        best = (d, foot, (float(Vb[i, 0]), float(Vb[i, 1])))
    if not math.isfinite(best[0]):
        D = Va[:, None, :] - Vb[None, :, :]
        d2 = np.einsum("nmj,nmj->nm", D, D)
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        best = (math.sqrt(float(d2[i, j])),
                (float(Va[i, 0]), float(Va[i, 1])),
                (float(Vb[j, 0]), float(Vb[j, 1])))
    return best


def test_the_gated_gap_is_the_ungated_gap_on_every_pair_shape():
    rng = np.random.default_rng(23)
    geoms = []
    for _ in range(150):
        x, y = rng.uniform(-300, 300), rng.uniform(-300, 300)
        kind = rng.integers(0, 3)
        if kind == 0:
            geoms.append(G.mask_geom(ring(x, y, rng.uniform(1, 60),
                                           rng.uniform(1, 60))))
        elif kind == 1:
            geoms.append(G.mask_geom(_way(x, y, rng.uniform(-80, 80),
                                          rng.uniform(-80, 80))))
        else:
            geoms.append(G.mask_geom([[(x, y)]]))
    # nested, touching and crossing pairs, on purpose
    geoms.append(G.mask_geom(ring(0, 0, 100, 100)))
    geoms.append(G.mask_geom(ring(10, 10, 5, 5)))
    geoms.append(G.mask_geom(ring(100, 0, 10, 10)))
    geoms.append(G.mask_geom(_way(-50, 50, 200, 0)))
    geoms.append(G.mask_geom(ring(20, -20, 10, 300)))
    checked = disjoint = 0
    for i in range(len(geoms)):
        for j in range(i, len(geoms)):
            got = G.gap_witness(geoms[i], geoms[j])
            want = _ungated(geoms[i], geoms[j])
            assert got == want, (i, j, got, want)
            checked += 1
            if G.bbox_gap(geoms[i], geoms[j]) > 0.0:
                disjoint += 1
    # vacuity floors: many pairs, both branches exercised
    assert checked > 10_000, checked
    assert 0 < disjoint < checked, (disjoint, checked)
