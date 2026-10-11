"""The stops an ``explore`` flies: ``SearchLoop.node_coverage_stops``."""
from __future__ import annotations

import numpy as np

from maps.search.tests.fakes import search_loop

_CELL_M = 2.0
# The mission's flight budget, the stop bound it passes.
_MAX_STOPS = 1000


def _cover(lp, fid, spacing_m, minor_spacing_m=1e9):
    """The stops as the mission asks for them; the default minor side is wider
    than any fixture, so the 1-D walk."""
    return lp.node_coverage_stops(fid, spacing_m, max_stops=_MAX_STOPS,
                                  minor_spacing_m=minor_spacing_m)


def _loop_over(cells):
    """A SearchLoop whose one node carries these raster cells."""
    class _Node:
        pass

    _Node.cells = np.array(cells)
    _Node.centroid = (cells[0][0], cells[0][1])
    return search_loop({"prior_9": _Node()}, cell_size_m=_CELL_M)


def _cell_centres(cells):
    """Where those raster cells sit in the world, rounded as a stop is."""
    return {(round(0.0 + (int(c[1]) + 0.5) * _CELL_M),
             round(0.0 + (int(c[0]) + 0.5) * _CELL_M)) for c in cells}


def _on_the_shape(stops, cells):
    return all(tuple(p) in _cell_centres(cells) for p in stops)


def _xs(cells):
    """The world x of each cell."""
    return sorted(round((int(c[1]) + 0.5) * _CELL_M) for c in cells)


# a 140 m road: one cell wide, 70 cells long, running along x
_ROAD = [[100, j] for j in range(50, 120)]


def test_every_stop_is_a_cell_of_the_thing_being_covered():
    lp = _loop_over(_ROAD)
    stops = _cover(lp, "prior_9", 20.0)
    assert stops, "a pass with no stops is not a pass"
    assert _on_the_shape(stops, _ROAD), stops
    # and they WALK the length rather than clustering at the middle
    assert max(p[0] for p in stops) - min(p[0] for p in stops) >= 100.0


def test_the_spacing_is_the_callers_and_it_sets_how_many_stops_there_are():
    """The caller passes the run's own observation extent, halved."""
    lp = _loop_over(_ROAD)
    # 140 m of road at 20 m a stop is 8; at 200 m it is one.
    assert len(_cover(lp, "prior_9", 20.0)) == 8
    assert len(_cover(lp, "prior_9", 70.0)) == 3
    assert len(_cover(lp, "prior_9", 200.0)) == 1


def test_a_shape_inside_one_observation_is_covered_from_its_MIDDLE():
    """k == 1 is a stop, not an end."""
    lp = _loop_over(_ROAD)
    one = _cover(lp, "prior_9", 500.0)
    assert len(one) == 1, one
    assert _on_the_shape(one, _ROAD), one
    xs = _xs(_ROAD)
    assert one[0][0] not in (xs[0], xs[-1]), (one, xs[0], xs[-1])
    assert abs(one[0][0] - (xs[0] + xs[-1]) / 2.0) <= _CELL_M, one


def test_the_stop_bound_binds_and_the_extremes_still_ride():
    """A thing longer than the bound is SAMPLED, and both ends are in it."""
    # So the fixture is deliberately absurd -- 8 km walked at 4 m -- because what is
    # pinned is that the guard still HOLDS and still keeps the ends, not that any street
    # will ever meet it.
    long_road = [[100, j] for j in range(0, 4000)]      # 8 km
    lp = _loop_over(long_road)
    stops = _cover(lp, "prior_9", 4.0)
    assert len(stops) == _MAX_STOPS, len(stops)
    assert _on_the_shape(stops, long_road), stops
    xs = _xs(long_road)
    assert stops[0][0] == xs[0] and stops[-1][0] == xs[-1], stops


def test_the_stops_are_in_shape_order_so_the_pass_flies_a_line():
    """An explore is a PASS."""
    lp = _loop_over(_ROAD)
    xs = [p[0] for p in _cover(lp, "prior_9", 20.0)]
    assert xs == sorted(xs) and xs[0] != xs[-1], xs


def test_one_representation_covers_polygon_linestring_and_point_alike():
    """The bake rasterizes everything into cells: a polygon fills, a LineString inflates
    to one cell of width (roads measure 1.00 cells per unit of length at the median
    on this bake), a node is one cell."""
    shapes = {
        "linestring": _ROAD,
        "polygon": [[r, c] for r in range(100, 108)
                    for c in range(50, 58)],              # a building
        "point": [[100, 50]],                             # a node
    }
    for name, cells in shapes.items():
        stops = _cover(_loop_over(cells), "prior_9", 20.0)
        assert stops, name
        assert _on_the_shape(stops, cells), name
        # integers, because a stop is spelled into a world-point token
        assert all(p == [int(p[0]), int(p[1])] for p in stops), name


def test_nothing_from_the_vector_ring_leaks_into_the_stops():
    cells = [[100, j] for j in range(50, 60)]
    lp = _loop_over(cells)
    lp.state.runtime_graph.nodes["prior_9"].world_contour = [
        [100.0, 200.0], [118.0, 200.0], [118.0, 202.0],
        [2500.0, 202.0], [2500.0, 900.0], [100.0, 202.0]]
    stops = _cover(lp, "prior_9", 20.0)
    assert stops and _on_the_shape(stops, cells), stops
    assert all(p[0] <= 120 for p in stops), stops       # the cells stop at 119


def test_a_fid_with_no_cells_has_nothing_to_cover():
    """``None``, which the caller reads as "this pass is the one stop it is" rather than
    as an empty cover it could mistake for a flown one."""
    lp = _loop_over(_ROAD)
    assert _cover(lp, "nope", 20.0) is None

    class _Bare:
        cells = np.array([])
        centroid = (0, 0)

    lp.state.runtime_graph.nodes["prior_9"] = _Bare()
    assert _cover(lp, "prior_9", 20.0) is None


def test_a_wide_shape_tiles_serpentine_at_the_short_side_pitch():
    """A block wider than the footprint's SHORT side in BOTH axes rows on the minor axis
    at that pitch, so the 1-D walk's under-covered middle is actually flown."""
    block = [[i, j] for i in range(0, 60) for j in range(0, 60)]  # 120x120 m
    lp = _loop_over(block)
    stops = _cover(lp, "prior_9", 20.0, minor_spacing_m=40.0)
    assert stops and _on_the_shape(stops, block), stops
    bands = sorted({int(p[1] // 40) for p in stops})
    assert len(bands) >= 3, stops       # 120 m of minor extent at 40 m pitch


def test_the_serpentine_rows_alternate_so_the_pass_stays_one_line():
    """Consecutive rows join at the near end: row 0 runs one way, row 1 the other."""
    block = [[i, j] for i in range(0, 60) for j in range(0, 60)]
    lp = _loop_over(block)
    stops = _cover(lp, "prior_9", 20.0, minor_spacing_m=40.0)
    rows: dict[int, list] = {}
    for p in stops:
        rows.setdefault(int(p[1] // 40), []).append(p[0])
    runs = [xs for _, xs in sorted(rows.items()) if len(xs) > 1]
    assert len(runs) >= 2, stops
    for a, b in zip(runs, runs[1:]):
        asc_a, asc_b = a == sorted(a), b == sorted(b)
        assert asc_a != asc_b, (a, b)


def test_a_narrow_shape_keeps_the_one_dimensional_walk_exactly():
    lp = _loop_over(_ROAD)
    assert (_cover(lp, "prior_9", 20.0, minor_spacing_m=240.0)
            == _cover(lp, "prior_9", 20.0))
