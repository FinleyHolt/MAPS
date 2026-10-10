"""The stops an ``explore`` flies: ``SearchLoop.node_coverage_stops``.

The geometry half of the covering pass. The loop half — the pass flying
them, the return leg, the one handle and the one picture — is pinned in
``maps/search/tests/test_code_surface_arm.py``.

THE ONE CONTRACT: every stop is ON the shape. It is asserted against the
node's OWN cells rather than through a distance predicate, so the serve and
the check share their DATA and not merely a rule. That matters here: the
vector ``world_contour`` is the UNCLIPPED OSM ring while cells are CLIPPED
to imagery coverage, and serving ring vertices once offered points nowhere
near the thing — 19% of contour priors, one of them 2.4 km off its own mask.
"""
from __future__ import annotations

import threading

import numpy as np

from maps.search.search_loop import SearchLoop, _COVER_MAX_STOPS

_CELL_M = 2.0


def _loop_over(cells):
    """A SearchLoop stand-in whose one node carries these raster cells.

    Cells arrive as an (N, 2) ARRAY on the runtime graph, where ``not
    cells`` raises "truth value of an array with more than one element is
    ambiguous" and takes the whole episode down as an unclassified
    no-prediction row. The first aimed arm died on exactly that with every
    fixture a plain list, so the suite could not see it: these are arrays.
    """
    class _Node:
        pass

    _Node.cells = np.array(cells)
    _Node.centroid = (cells[0][0], cells[0][1])

    class _Graph:
        nodes = {"prior_9": _Node()}

    class _S:
        runtime_graph = None

    class _Loop(SearchLoop):     # read-only properties on the real loop, so
        _cell_size_m = _CELL_M   # the stand-in overrides the class
        _tile_origin_xy = (0.0, 0.0)
        state = _S()

    lp = _Loop.__new__(_Loop)
    lp._lock = threading.RLock()
    lp._graph = _Graph()
    return lp


def _cell_centres(cells):
    """Where those raster cells sit in the world, rounded as a stop is."""
    return {(round(0.0 + (int(c[1]) + 0.5) * _CELL_M),
             round(0.0 + (int(c[0]) + 0.5) * _CELL_M)) for c in cells}


def _on_the_shape(stops, cells):
    return all(tuple(p) in _cell_centres(cells) for p in stops)


def _xs(cells):
    """The world x of each cell. A cell is ``[row, col]`` and x comes off
    the COLUMN, so a road spelled by varying the second index runs along
    x — which is the axis these stops walk."""
    return sorted(round((int(c[1]) + 0.5) * _CELL_M) for c in cells)


# a 140 m road: one cell wide, 70 cells long, running along x
_ROAD = [[100, j] for j in range(50, 120)]


def test_every_stop_is_a_cell_of_the_thing_being_covered():
    lp = _loop_over(_ROAD)
    stops = lp.node_coverage_stops("prior_9", 20.0)
    assert stops, "a pass with no stops is not a pass"
    assert _on_the_shape(stops, _ROAD), stops
    # and they WALK the length rather than clustering at the middle
    assert max(p[0] for p in stops) - min(p[0] for p in stops) >= 100.0


def test_the_spacing_is_the_callers_and_it_sets_how_many_stops_there_are():
    """The caller passes the run's own observation extent, halved.

    A number written down twice goes stale on one side: a bake or an
    altitude change moves the footprint, and a constant here would keep
    covering the shape at the extent of an observation that is gone.
    """
    lp = _loop_over(_ROAD)
    # 140 m of road at 20 m a stop is 8; at 200 m it is one.
    assert len(lp.node_coverage_stops("prior_9", 20.0)) == 8
    assert len(lp.node_coverage_stops("prior_9", 70.0)) == 3
    assert len(lp.node_coverage_stops("prior_9", 200.0)) == 1


def test_a_shape_inside_one_observation_is_covered_from_its_MIDDLE():
    """k == 1 is a stop, not an end.

    Exploring a shed is one flight and not a tour of one — and standing on
    the far end of a thing that fits in the view square sees half of it.
    """
    lp = _loop_over(_ROAD)
    one = lp.node_coverage_stops("prior_9", 500.0)
    assert len(one) == 1, one
    assert _on_the_shape(one, _ROAD), one
    xs = _xs(_ROAD)
    assert one[0][0] not in (xs[0], xs[-1]), (one, xs[0], xs[-1])
    assert abs(one[0][0] - (xs[0] + xs[-1]) / 2.0) <= _CELL_M, one


def test_the_stop_bound_binds_and_the_extremes_still_ride():
    """A thing longer than the bound is SAMPLED, and both ends are in it.

    The bound is reported by the caller, never silent (the pass's note says
    what fraction it reached). What it costs is DENSITY, not span: the walk
    still runs end to end, so a truncated pass covers the whole shape
    thinly rather than covering the first part of it well.
    """
    # The backstop is a RUNAWAY guard now, not a budget: reaching it takes
    # a shape no real feature approaches (the bake's longest road is 19 km,
    # which at the observation's own ~160 m spacing wants ~120 stops). So
    # the fixture is deliberately absurd -- 8 km walked at 4 m -- because
    # what is pinned is that the guard still HOLDS and still keeps the ends,
    # not that any street will ever meet it.
    long_road = [[100, j] for j in range(0, 4000)]      # 8 km
    lp = _loop_over(long_road)
    stops = lp.node_coverage_stops("prior_9", 4.0)
    assert len(stops) == _COVER_MAX_STOPS, len(stops)
    assert _on_the_shape(stops, long_road), stops
    xs = _xs(long_road)
    assert stops[0][0] == xs[0] and stops[-1][0] == xs[-1], stops


def test_the_stops_are_in_shape_order_so_the_pass_flies_a_line():
    """An explore is a PASS. Stops in shape order fly a line; stops in any
    other order fly a scribble over the same ground for the same money."""
    lp = _loop_over(_ROAD)
    xs = [p[0] for p in lp.node_coverage_stops("prior_9", 20.0)]
    assert xs == sorted(xs) and xs[0] != xs[-1], xs


def test_one_representation_covers_polygon_linestring_and_point_alike():
    """The bake rasterizes everything into cells: a polygon fills, a
    LineString inflates to one cell of width (roads measure 1.00 cells per
    unit of length at the median on this bake), a node is one cell. All
    6188 priors carry them; only 64% carry a vector ring. Deriving the
    stops from the universal form is what removes the special cases — and
    with them the clipped-vs-unclipped divergence they introduced.
    """
    shapes = {
        "linestring": _ROAD,
        "polygon": [[r, c] for r in range(100, 108)
                    for c in range(50, 58)],              # a building
        "point": [[100, 50]],                             # a node
    }
    for name, cells in shapes.items():
        stops = _loop_over(cells).node_coverage_stops("prior_9", 20.0)
        assert stops, name
        assert _on_the_shape(stops, cells), name
        # integers, because a stop is spelled into a world-point token
        assert all(p == [int(p[0]), int(p[1])] for p in stops), name


def test_nothing_from_the_vector_ring_leaks_into_the_stops():
    """This node's ring runs 2.4 km past its own cells — the real shape of
    the clipping mismatch. Not one stop may come from it."""
    cells = [[100, j] for j in range(50, 60)]
    lp = _loop_over(cells)
    lp._graph.nodes["prior_9"].world_contour = [
        [100.0, 200.0], [118.0, 200.0], [118.0, 202.0],
        [2500.0, 202.0], [2500.0, 900.0], [100.0, 202.0]]
    stops = lp.node_coverage_stops("prior_9", 20.0)
    assert stops and _on_the_shape(stops, cells), stops
    assert all(p[0] <= 120 for p in stops), stops       # the cells stop at 119


def test_a_fid_with_no_cells_has_nothing_to_cover():
    """``None``, which the caller reads as "this pass is the one stop it
    is" rather than as an empty cover it could mistake for a flown one."""
    lp = _loop_over(_ROAD)
    assert lp.node_coverage_stops("nope", 20.0) is None

    class _Bare:
        cells = np.array([])
        centroid = (0, 0)

    lp._graph.nodes["prior_9"] = _Bare()
    assert lp.node_coverage_stops("prior_9", 20.0) is None


def test_a_wide_shape_tiles_serpentine_at_the_short_side_pitch():
    """A block wider than the footprint's SHORT side in BOTH axes rows on
    the minor axis at that pitch, so the 1-D walk's under-covered middle
    is actually flown. Every stop is still a cell of the thing."""
    block = [[i, j] for i in range(0, 60) for j in range(0, 60)]  # 120x120 m
    lp = _loop_over(block)
    stops = lp.node_coverage_stops("prior_9", 20.0, minor_spacing_m=40.0)
    assert stops and _on_the_shape(stops, block), stops
    bands = sorted({int(p[1] // 40) for p in stops})
    assert len(bands) >= 3, stops       # 120 m of minor extent at 40 m pitch


def test_the_serpentine_rows_alternate_so_the_pass_stays_one_line():
    """Consecutive rows join at the near end: row 0 runs one way, row 1
    the other. Same-direction rows would fly a carriage return between
    every pair for the same money."""
    block = [[i, j] for i in range(0, 60) for j in range(0, 60)]
    lp = _loop_over(block)
    stops = lp.node_coverage_stops("prior_9", 20.0, minor_spacing_m=40.0)
    rows: dict[int, list] = {}
    for p in stops:
        rows.setdefault(int(p[1] // 40), []).append(p[0])
    runs = [xs for _, xs in sorted(rows.items()) if len(xs) > 1]
    assert len(runs) >= 2, stops
    for a, b in zip(runs, runs[1:]):
        asc_a, asc_b = a == sorted(a), b == sorted(b)
        assert asc_a != asc_b, (a, b)


def test_a_narrow_shape_keeps_the_one_dimensional_walk_exactly():
    """The serpentine arms only past the pitch in BOTH axes: a road is
    1-D by nature and its stops must not move when the caller starts
    passing the short side."""
    lp = _loop_over(_ROAD)
    assert (lp.node_coverage_stops("prior_9", 20.0, minor_spacing_m=240.0)
            == lp.node_coverage_stops("prior_9", 20.0))
