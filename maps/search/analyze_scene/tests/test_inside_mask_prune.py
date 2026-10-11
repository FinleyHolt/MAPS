import numpy as np
import pytest

from maps.search.analyze_scene import geom_compute as G


def _unpruned(P, A, B):
    out = np.zeros(len(P), dtype=bool)
    if len(P) == 0 or len(A) == 0:
        return out
    Ax, Ay = A[:, 0][None, :], A[:, 1][None, :]
    Bx, By = B[:, 0][None, :], B[:, 1][None, :]
    px, py = P[:, 0][:, None], P[:, 1][:, None]
    cond = (Ay > py) != (By > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xin = (Bx - Ax) * (py - Ay) / (By - Ay) + Ax
    return ((cond & (px < xin)).sum(axis=1) & 1).astype(bool)


def _ring(x, y, w, h):
    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
    return [(pts[i], pts[i + 1]) for i in range(4)]


def _arrays(edges):
    return (np.asarray([e[0] for e in edges], dtype=float),
            np.asarray([e[1] for e in edges], dtype=float))


def test_the_prune_is_exact_on_the_boundary_cases_a_bound_gets_wrong():
    """Horizontal edges, and points sharing a y with a vertex."""
    edges = _ring(0.0, 0.0, 10.0, 10.0) + _ring(2.0, 2.0, 3.0, 3.0)
    A, B = _arrays(edges)
    ys = [0.0, 2.0, 5.0, 10.0, -1.0, 11.0]
    xs = [-1.0, 0.0, 1.0, 3.0, 5.0, 10.0, 11.0]
    P = np.asarray([(x, y) for y in ys for x in xs], dtype=float)
    assert np.array_equal(G._inside_mask_arr(P, A, B), _unpruned(P, A, B))


def test_a_hole_still_subtracts():
    """Parity over all loops together is the membership rule; a prune that dropped one
    loop's edges would silently fill the hole."""
    outer = _ring(0.0, 0.0, 100.0, 100.0)
    hole = _ring(40.0, 40.0, 20.0, 20.0)
    A, B = _arrays(outer + hole)
    P = np.asarray([(50.0, 50.0), (10.0, 50.0), (50.0, 10.0)], dtype=float)
    got = G._inside_mask_arr(P, A, B)
    assert np.array_equal(got, _unpruned(P, A, B))
    assert not bool(got[0]), "the hole must not read as inside"
    assert bool(got[1]) and bool(got[2])


@pytest.mark.parametrize("n_points", [0, 1, 2])
def test_the_degenerate_inputs_answer_the_same(n_points):
    A, B = _arrays(_ring(0.0, 0.0, 4.0, 4.0))
    P = np.asarray([(1.0, 1.0)] * n_points, dtype=float).reshape(-1, 2)
    assert np.array_equal(G._inside_mask_arr(P, A, B), _unpruned(P, A, B))
    empty = np.zeros((0, 2), dtype=float)
    assert not G._inside_mask_arr(P, empty, empty).any()


def test_more_than_one_chunk_is_actually_exercised(monkeypatch):
    """The prune is per chunk, so a test that fits in one chunk proves nothing about it."""
    rng = np.random.default_rng(3)
    edges = []
    for _ in range(200):
        edges += _ring(rng.uniform(0, 500), rng.uniform(0, 500),
                       rng.uniform(5, 30), rng.uniform(5, 30))
    A, B = _arrays(edges)
    P = np.stack([rng.uniform(0, 500, 400), rng.uniform(0, 500, 400)], 1)
    expected = _unpruned(P, A, B)
    monkeypatch.setattr(G, "_PAIR_CHUNK", len(A) * 4)
    got = G._inside_mask_arr(P, A, B)
    assert np.array_equal(got, expected)
    assert 1 <= int(expected.sum()) < len(P)
