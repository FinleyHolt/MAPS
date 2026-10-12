"""The cardinal frame, pinned on five shapes."""

import numpy as np
import pytest

from maps.search.analyze_scene.geom_compute import CARDINALS, cardinal_offset


def road_ns(x=0.0, y0=-500.0, y1=500.0, n=501):
    return np.column_stack([np.full(n, x), np.linspace(y0, y1, n)])


def test_straight_road_signs_both_ways():
    # A positive return means the point IS in the direction asked about,
    # and the magnitude is its clearance past the reference.
    road = road_ns()
    assert cardinal_offset(road, (-80.0, 0.0), "west") == 80.0
    assert cardinal_offset(road, (-80.0, 0.0), "east") == -80.0


def test_self_approaching_road_abstains_where_min_gap_inverts():
    """THE counterexample: a U-shaped road with the point between its legs."""
    u_road = np.vstack([road_ns(), road_ns(x=-200.0),
                        np.column_stack([np.linspace(-200, 0, 201),
                                         np.full(201, 500.0)])])
    inside = (-150.0, 0.0)

    d = np.hypot(u_road[:, 0] - inside[0], u_road[:, 1] - inside[1])
    assert inside[0] - u_road[int(d.argmin())][0] > 0, "min-gap must invert here"

    assert cardinal_offset(u_road, inside, "east") is None
    assert cardinal_offset(u_road, inside, "west") is None
    # A point clear of the whole U still decides, and decides west.
    assert cardinal_offset(u_road, (-400.0, 0.0), "west") == 200.0


def test_east_west_road_asked_east_abstains():
    # The road's normal is north-south, so "east of" projects to nothing.
    # Genuine ambiguity, not imprecision -- and the orthogonal query works.
    ew = np.column_stack([np.linspace(-500, 500, 501), np.zeros(501)])
    assert cardinal_offset(ew, (0.0, 200.0), "east") is None
    assert cardinal_offset(ew, (0.0, 200.0), "north") == 200.0


def test_closed_loop_inside_abstains_outside_decides():
    th = np.linspace(0, 2 * np.pi, 361)
    ring = np.column_stack([300 * np.cos(th), 300 * np.sin(th)])
    assert all(cardinal_offset(ring, (0.0, 0.0), d) is None for d in CARDINALS)
    assert cardinal_offset(ring, (500.0, 0.0), "east") == 200.0


def test_cul_de_sac_bulb_abstains_stem_decides():
    th = np.linspace(0, 2 * np.pi, 361)
    bulb = np.vstack([
        np.column_stack([np.zeros(200), np.linspace(-400, 0, 200)]),
        np.column_stack([100 * np.cos(th), 100 * np.sin(th) + 100])])
    assert cardinal_offset(bulb, (0.0, 100.0), "east") is None
    assert cardinal_offset(bulb, (-300.0, -200.0), "west") == 300.0


def test_isotropic_blob_has_no_axis_and_still_decides():
    # PCA has nothing to say about a blob; the band does not need an axis.
    blob = np.random.default_rng(0).normal(0, 40, size=(2000, 2))
    east = cardinal_offset(blob, (400.0, 0.0), "east")
    assert east is not None and 200.0 < east < 400.0


def test_empty_reference_abstains_and_a_diagonal_refuses():
    # Two different failures, distinguishable: no geometry vs no such axis.
    assert cardinal_offset(np.empty((0, 2)), (0.0, 0.0), "east") is None
    with pytest.raises(ValueError):
        cardinal_offset(road_ns(), (0.0, 0.0), "northeast")


def test_band_width_is_not_load_bearing(monkeypatch):
    from maps.search.analyze_scene import geom_compute
    road = road_ns()
    got = set()
    for w in (50.0, 150.0, 1000.0):
        monkeypatch.setattr(geom_compute, "BAND_M", w)
        got.add(cardinal_offset(road, (-80.0, 0.0), "west"))
    assert got == {80.0}
