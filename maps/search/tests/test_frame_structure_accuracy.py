"""Does the measured pool axis match the axis a lot ACTUALLY has?

``_row_axis_direction`` (+ ``_row_partition_tightness``) is the shared
estimator behind ``frame:{reference:'auto'}`` — the pool's own axis that
vx_/vy_ project against. The lots here are generated from a known
(rows x cols, pitch, rotation) so the correct axis is known by construction,
and the assertions are about accuracy against that truth.

It found, on the first run, that a regular lot deeper than it is wide got a
PCA axis 90 degrees wrong — see ``test_axis_tie_on_a_regular_lattice_is_exact``.
"""
from __future__ import annotations

import math
import random

import pytest

from maps.search.analyze_scene.geom_compute import (
    _row_axis_direction,
    _row_partition_tightness,
)


# A CityNav parking row: cars ~5 m apart along the row, rows ~12 m apart.
PITCH_M = 5.0
ROW_PITCH_M = 12.0


def lot(n_rows, n_cols, rot_deg=0.0, jitter_m=0.0, seed=0,
        pitch=PITCH_M, row_pitch=ROW_PITCH_M, drop=()):
    """A regular lot rotated by ``rot_deg``, with the truth known by build.

    Each point carries its build-truth ``row``/``col``.
    """
    rng = random.Random(seed)
    th = math.radians(rot_deg)
    pts = []
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in drop:
                continue
            x = c * pitch + rng.uniform(-jitter_m, jitter_m)
            y = r * row_pitch + rng.uniform(-jitter_m, jitter_m)
            pts.append({"fid": f"r{r}c{c}", "class": "car", "row": r, "col": c,
                        "xy": [x * math.cos(th) - y * math.sin(th),
                               x * math.sin(th) + y * math.cos(th)]})
    return pts


def truth_partition(n_rows, n_cols, drop=()):
    return {frozenset(f"r{r}c{c}" for c in range(n_cols) if (r, c) not in drop)
            for r in range(n_rows)}


def test_axis_tie_on_a_regular_lattice_is_exact():
    """Both candidate axes score EXACTLY zero, so the tie-break decides.

    ``_row_axis_direction`` scores the NN axis and a whole-cluster PCA axis by
    the within-row spread each would produce, and takes the tighter. On a
    regular lattice BOTH partitions are perfectly tight, so the comparison
    ``sc < best_score - 1e-6`` never fires and ``best`` keeps its initial value
    — the tie must fall to NN (the measured local lattice), never PCA.
    """
    P = [tuple(d["xy"]) for d in lot(3, 5)]
    along = _row_partition_tightness(P, (1.0, 0.0))   # the true row direction
    across = _row_partition_tightness(P, (0.0, 1.0))
    assert along == across == 0.0, (along, across)

    u, conf, src = _row_axis_direction(P)
    assert src == "nn" and conf > 0.0
    assert abs(abs(u[0]) - 1.0) < 1e-6 and abs(u[1]) < 1e-6, u


@pytest.mark.parametrize("rot", [0, 10, 20, 30, 45, 60, 90])
def test_rotated_lot_recovers_its_axis(rot):
    """The recovered axis tracks the build rotation (mod 180: an axis is a
    direction pair)."""
    P = [tuple(d["xy"]) for d in lot(3, 5, rot_deg=rot)]
    u, _conf, _src = _row_axis_direction(P)
    got = math.degrees(math.atan2(u[1], u[0])) % 180.0
    want = rot % 180.0
    diff = min(abs(got - want), 180.0 - abs(got - want))
    assert diff < 2.0, (rot, got, u)


@pytest.mark.parametrize("n_rows,n_cols", [(2, 6), (3, 8), (3, 5), (5, 10)])
def test_axis_correct_for_wide_and_deep_lots(n_rows, n_cols):
    """Deeper-than-wide lots are the PCA failure case (its major axis is the
    ACROSS-row one there); the estimator must still recover the row axis."""
    P = [tuple(d["xy"]) for d in lot(n_rows, n_cols)]
    u, _conf, _src = _row_axis_direction(P)
    assert abs(abs(u[0]) - 1.0) < 1e-6 and abs(u[1]) < 1e-6, (n_rows, n_cols, u)


def test_axis_is_stable_across_positional_jitter():
    """Sub-metre noise must not flip the recovered axis (the tile bake is
    2 m)."""
    for jit in (0.0, 0.1, 0.3, 0.8):
        P = [tuple(d["xy"]) for d in lot(3, 5, jitter_m=jit, seed=7)]
        u, _conf, _src = _row_axis_direction(P)
        got = math.degrees(math.atan2(u[1], u[0])) % 180.0
        assert min(got, 180.0 - got) < 10.0, (jit, got)


def test_a_pool_with_no_local_lattice_falls_back_to_pca():
    """Every pairwise gap above ``nn_radius_m`` starves the NN estimator, so
    the axis falls back to PCA with conf 0.0 — a guess must announce itself."""
    P = [(0.0, 0.0), (20.0, 3.0), (41.0, -2.0), (63.0, 5.0),
         (12.0, 28.0), (48.0, 24.0)]
    _u, conf, src = _row_axis_direction(P)
    assert src == "pca" and conf == 0.0


# --- vacuity floor ----------------------------------------------------------

def test_the_generator_actually_builds_what_it_claims():
    """If `lot` silently produced one row, every assertion above would pass
    while testing nothing. (`lot`/`truth_partition` also feed the dashboard
    lab's truth overlay.)"""
    pts = lot(3, 5, rot_deg=30)
    assert len(pts) == 15
    xs = {round(p["xy"][0], 6) for p in pts}
    assert len(xs) > 5, "a rotated lot must not collapse onto shared columns"
    assert len(truth_partition(3, 5)) == 3
