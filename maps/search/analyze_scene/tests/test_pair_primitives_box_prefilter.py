"""The crossing and containment primitives prefilter by box, exactly.

A segment wholly outside the other set's bounding box crosses nothing in
it, and a point outside a set of closed loops' box crosses an even number
of their edges. Both prefilters are checked against the unfiltered bodies
on random geometry built to include touches, vertices on the box edge,
nested loops and a point-mask, so the answer -- the first crossing pair's
point, and every point's parity -- is the unfiltered one.
"""
import numpy as np

from maps.search.analyze_scene import geom_compute as G


def _cross_unfiltered(A1, B1, A2, B2):
    if len(A1) == 0 or len(A2) == 0:
        return None
    r = B1 - A1
    s = B2 - A2
    step = max(1, int(G._PAIR_CHUNK // max(1, len(A2))))
    for k in range(0, len(A1), step):
        a1, b1, rr = A1[k:k + step], B1[k:k + step], r[k:k + step]
        W1 = A2[None, :, :] - a1[:, None, :]
        W2 = B2[None, :, :] - a1[:, None, :]
        o1 = rr[:, None, 0] * W1[..., 1] - rr[:, None, 1] * W1[..., 0]
        o2 = rr[:, None, 0] * W2[..., 1] - rr[:, None, 1] * W2[..., 0]
        W3 = a1[:, None, :] - A2[None, :, :]
        W4 = b1[:, None, :] - A2[None, :, :]
        o3 = s[None, :, 0] * W3[..., 1] - s[None, :, 1] * W3[..., 0]
        o4 = s[None, :, 0] * W4[..., 1] - s[None, :, 1] * W4[..., 0]
        hit = (o1 * o2 < 0.0) & (o3 * o4 < 0.0)
        if hit.any():
            i, j = np.unravel_index(int(np.argmax(hit)), hit.shape)
            rxs = rr[i, 0] * s[j, 1] - rr[i, 1] * s[j, 0]
            t = (W1[i, j, 0] * s[j, 1] - W1[i, j, 1] * s[j, 0]) / rxs
            return (float(a1[i, 0] + t * rr[i, 0]),
                    float(a1[i, 1] + t * rr[i, 1]))
    return None


def _inside_unfiltered(P, A, B):
    out = np.zeros(len(P), dtype=bool)
    if len(P) == 0 or len(A) == 0:
        return out
    ylo = np.minimum(A[:, 1], B[:, 1])
    yhi = np.maximum(A[:, 1], B[:, 1])
    order = np.argsort(P[:, 1], kind="stable")
    step = max(1, int(G._PAIR_CHUNK // max(1, len(A))))
    for s in range(0, len(order), step):
        idx = order[s:s + step]
        p = P[idx]
        lo, hi = float(p[:, 1].min()), float(p[:, 1].max())
        sel = (ylo <= hi) & (yhi > lo)
        if not sel.any():
            continue
        a, b = A[sel], B[sel]
        Ax, Ay = a[:, 0][None, :], a[:, 1][None, :]
        Bx, By = b[:, 0][None, :], b[:, 1][None, :]
        px, py = p[:, 0][:, None], p[:, 1][:, None]
        cond = (Ay > py) != (By > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (Bx - Ax) * (py - Ay) / (By - Ay) + Ax
        cross = cond & (px < xin)
        out[idx] = (cross.sum(axis=1) & 1).astype(bool)
    return out


def _ring(x, y, w, h):
    return [[(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]]


def _shapes(rng, n):
    out = []
    for _ in range(n):
        x, y = rng.uniform(-200, 200), rng.uniform(-200, 200)
        k = rng.integers(0, 3)
        if k == 0:
            out.append(G.mask_geom(_ring(x, y, rng.uniform(1, 50),
                                         rng.uniform(1, 50))))
        elif k == 1:
            out.append(G.mask_geom([[(x, y), (x + rng.uniform(-90, 90),
                                             y + rng.uniform(-90, 90))]]))
        else:
            out.append(G.mask_geom([[(x, y)]]))
    # a big ring with a hole, a touching ring, a long way across everything
    out.append(G.mask_geom(_ring(-100, -100, 200, 200) + _ring(-20, -20, 40, 40)))
    out.append(G.mask_geom(_ring(100, -100, 30, 30)))
    out.append(G.mask_geom([[(-300, 0), (300, 0)]]))
    return out


def test_segments_cross_matches_the_unfiltered_first_hit():
    rng = np.random.default_rng(41)
    shapes = _shapes(rng, 120)
    hits = checked = 0
    for i in range(len(shapes)):
        for j in range(len(shapes)):
            a, b = shapes[i], shapes[j]
            got = G._segments_cross(a["A"], a["B"], b["A"], b["B"])
            want = _cross_unfiltered(a["A"], a["B"], b["A"], b["B"])
            assert got == want, (i, j, got, want)
            checked += 1
            hits += want is not None
    assert checked > 10_000 and 0 < hits < checked, (checked, hits)


def test_inside_mask_matches_the_unfiltered_parity_including_the_box_edge():
    rng = np.random.default_rng(42)
    shapes = [s for s in _shapes(rng, 60) if len(s["CA"])]
    assert len(shapes) > 10
    # points scattered everywhere, PLUS points on every loop's box edges and
    # corners, where an off-by-one in the prefilter would show
    P = rng.uniform(-260, 260, size=(3000, 2))
    edge = []
    for s in shapes:
        x0, y0, x1, y1 = G.geom_bbox(s)
        for x in (x0, x1, (x0 + x1) / 2):
            for y in (y0, y1, (y0 + y1) / 2):
                edge.append([x, y])
    P = np.concatenate([P, np.asarray(edge)])
    inside = 0
    for s in shapes:
        got = G._inside_mask_arr(P, s["CA"], s["CB"])
        want = _inside_unfiltered(P, s["CA"], s["CB"])
        assert np.array_equal(got, want)
        inside += int(want.sum())
    assert inside > 100, inside
