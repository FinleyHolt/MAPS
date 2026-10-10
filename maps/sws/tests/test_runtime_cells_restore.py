"""A tracked detection stores its projected footprint on ``node.cells``, the
``(N, 2) int32`` format prebake nodes carry, re-stamped every frame.
Re-identification is the exact ``track_id``: the same id folds onto its node,
a different id mints a new one.
"""

from __future__ import annotations

import numpy as np

from maps.sws.graph import upsert_tracked_detected
from maps.sws.graph.builder import build_object_graph
from maps.sws.graph.model import MapObject


def _seed_graph():
    feats = {"buildings": [{
        "name": "bldg_0",
        "cells": [[r, c] for r in range(5, 9) for c in range(5, 9)],
        "centroid_row": 7, "centroid_col": 7,
    }]}
    g = build_object_graph(feats, cell_size_m=2.0)
    return g


def _det_node(track_id):
    return MapObject(
        id="detected_pending", class_name="car", name=None, tags={},
        centroid=(0, 0), bbox=(0, 0, 0, 0), cell_count=1,
        orientation=None, orientation_confidence=0.0, provenance="detected",
        confidence=0.8, first_seen_t=0.0, last_seen_t=0.0, observation_count=0,
    )


def _square_mask(grid, r0, c0, side):
    m = np.zeros(grid, dtype=bool)
    m[r0:r0 + side, c0:c0 + side] = True
    return m


def _cells_of(mask):
    rs, cs = np.where(mask)
    return {(int(r), int(c)) for r, c in zip(rs.tolist(), cs.tolist())}


def _sparse(mask):
    """The projector's sparse output shape for a dense fixture mask."""
    rs, cs = np.where(mask)
    return {"rows": rs.astype(np.int32), "cols": cs.astype(np.int32)}


def test_fresh_detection_stores_cells_matching_projection():
    grid = (32, 32)
    g = _seed_graph()
    mask = _square_mask(grid, 10, 12, 3)
    fid = upsert_tracked_detected(
        g, track_id="t1", base_node=_det_node("t1"), **_sparse(mask),
        confidence=0.8, now=0.0,
    )
    node = g.nodes[fid]
    assert node.cells is not None
    assert node.cells.dtype == np.int32
    assert node.cells.shape == (9, 2)                      # 3x3 mask
    assert _cells_of(mask) == {(int(r), int(c)) for r, c in node.cells}
    # cells agree with the derived centroid/cell_count (same projection source).
    assert node.cell_count == 9
    assert node.cells.shape[0] == node.cell_count


def test_repeat_sighting_restamps_cells_and_keeps_one_node():
    grid = (32, 32)
    g = _seed_graph()
    m0 = _square_mask(grid, 10, 12, 3)
    fid0 = upsert_tracked_detected(
        g, track_id="t1", base_node=_det_node("t1"), **_sparse(m0),
        confidence=0.8, now=0.0,
    )
    # Same track_id, a moved/grown footprint next frame.
    m1 = _square_mask(grid, 14, 16, 4)
    fid1 = upsert_tracked_detected(
        g, track_id="t1", base_node=_det_node("t1"), **_sparse(m1),
        confidence=0.9, now=1.0,
    )
    assert fid1 == fid0                                    # exact track_id re-id
    node = g.nodes[fid0]
    assert node.observation_count == 2
    assert _cells_of(m1) == {(int(r), int(c)) for r, c in node.cells}  # re-stamped


def test_distinct_track_ids_mint_distinct_nodes_each_with_cells():
    grid = (32, 32)
    g = _seed_graph()
    m1 = _square_mask(grid, 5, 5, 2)
    m2 = _square_mask(grid, 20, 20, 3)
    f1 = upsert_tracked_detected(
        g, track_id="t1", base_node=_det_node("t1"), **_sparse(m1),
        confidence=0.8, now=0.0,
    )
    f2 = upsert_tracked_detected(
        g, track_id="t2", base_node=_det_node("t2"), **_sparse(m2),
        confidence=0.8, now=0.0,
    )
    assert f1 != f2                                        # track_id is the sole re-id
    assert g.nodes[f1].cells is not None and g.nodes[f2].cells is not None
    assert _cells_of(m1) == {(int(r), int(c)) for r, c in g.nodes[f1].cells}
    assert _cells_of(m2) == {(int(r), int(c)) for r, c in g.nodes[f2].cells}
