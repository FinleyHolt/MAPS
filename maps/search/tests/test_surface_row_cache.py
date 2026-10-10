"""An unchanged node is not rebuilt, and a changed one always is.

The surface rebuild is a pure function of node content and tile geometry, and
it runs once per flight over every node on the graph. An ``explore`` runs it
once per stop: calcity::6#6 rebuilt all 6,188 Cal City rows 262 times while
its graph gained nothing on any of those stops. Reuse is worth 2.66 s -> 0.21 s
there.

The cache keys on CONTENT because neither of the cheaper signals is sound
here. Identity fails outright: the ingest fold rebinds ``runtime_graph`` to the
server's snapshot, so every node object is new on every stop. A hand-kept
dirty-fid set fails silently and later: it has to be right about every
mutation path in the tree forever, and the first one anyone forgets serves a
stale row with nothing saying so.

These tests are the equivalence the speed rests on: cached and uncached
surfaces must be EQUAL, including after a mutation, including after the node
objects are replaced wholesale the way a fold replaces them.
"""

from __future__ import annotations

import numpy as np

from maps.search.analyze_scene import geom_program as gp


class _Node:
    def __init__(self, r, c, name=None, provenance="prebaked", cls="building"):
        self.cells = np.asarray([[r + i, c + j] for i in range(3)
                                 for j in range(4)], dtype=np.int32)
        self.world_xy = [float(c) * 2.0, float(r) * 2.0]
        self.centroid = (int(r), int(c))
        self.centerline = None
        self.class_name = cls
        self.name = name
        self.provenance = provenance
        self.tags = {"source": "osm"}


def _graph(n=9, name_suffix=""):
    return {f"building_{i}": _Node(10 * i, 10 * i, name=f"b{i}{name_suffix}")
            for i in range(1, n + 1)}


def _build(nodes):
    return gp.build_surface_from_graph(nodes.items(), (0.0, 0.0), 2.0)


def _uncached(nodes):
    gp._ROW_CACHE.clear()
    out = _build(nodes)
    gp._ROW_CACHE.clear()
    return out


def test_a_warm_cache_returns_the_same_surface_as_a_cold_one():
    nodes = _graph()
    ref = _uncached(nodes)
    cold = _build(nodes)
    warm = _build(nodes)
    assert cold == ref
    assert warm == ref
    # vacuity floor: an empty surface would satisfy every equality above
    assert len(ref[1]) == 9, len(ref[1])


def test_a_mutated_node_is_rebuilt():
    nodes = _graph()
    before = _build(nodes)
    nodes["building_3"].name = "renamed"
    after = _build(nodes)
    assert after != before
    assert after == _uncached(nodes)
    changed = [a for a in after[1] if a["fid"] == "building_3"]
    assert changed and changed[0]["name"] == "renamed", changed


def test_replacing_every_node_object_still_hits_the_cache():
    """What the ingest fold does: same content, all-new objects.

    An identity-keyed cache would miss on all of them -- which is the whole
    reason this one hashes content.
    """
    ref = _uncached(_graph())
    gp._ROW_CACHE.clear()
    _build(_graph())
    n_after_first = len(gp._ROW_CACHE)
    _build(_graph())  # a wholly separate set of node objects, same content
    assert len(gp._ROW_CACHE) == n_after_first, (
        n_after_first, len(gp._ROW_CACHE))
    assert _build(_graph()) == ref


def test_a_changed_cells_array_is_not_reused():
    """Cells are the bulk of a node and the thing hashed by bytes."""
    nodes = _graph()
    _build(nodes)
    nodes["building_2"].cells = np.asarray([[500, 500], [500, 501]],
                                           dtype=np.int32)
    assert _build(nodes) == _uncached(nodes)


def test_the_cache_does_not_leak_the_rows_it_hands_back():
    """Callers stamp rows (``covers_window``, ``end0_xy``); the cached
    original must stay the pre-stamp row, or the second rebuild inherits the
    first one's stamps."""
    nodes = _graph()
    first = _build(nodes)
    for row in first[1]:
        row["covers_window"] = True
        row["scribbled"] = 1
    second = _build(nodes)
    assert all("scribbled" not in r for r in second[1])
    assert second == _uncached(nodes)


def test_geometry_is_part_of_the_key():
    """The same node under a different origin or cell size is a different
    row, so the geometry cannot be left out of the key."""
    nodes = _graph()
    a = gp.build_surface_from_graph(nodes.items(), (0.0, 0.0), 2.0)
    b = gp.build_surface_from_graph(nodes.items(), (1000.0, 0.0), 2.0)
    assert a != b
    gp._ROW_CACHE.clear()
    assert b == gp.build_surface_from_graph(nodes.items(), (1000.0, 0.0), 2.0)


def test_overflow_drops_the_cache_rather_than_growing_without_bound():
    original = gp._ROW_CACHE_MAX
    gp._ROW_CACHE.clear()
    try:
        gp._ROW_CACHE_MAX = 4
        nodes = _graph(n=6)
        out = _build(nodes)
        assert len(gp._ROW_CACHE) <= 6, len(gp._ROW_CACHE)
        assert out == _uncached(nodes)
    finally:
        gp._ROW_CACHE_MAX = original
        gp._ROW_CACHE.clear()
