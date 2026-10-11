"""The tracked upsert: one node per exact ``track_id``, its footprint re-stamped on ``node.cells``
(``(N, 2) int32``, the prebake format) at every sighting."""

from __future__ import annotations

import numpy as np

from maps.sws.graph.builder import build_object_graph
from maps.sws.graph.model import MapObject
from maps.sws.graph.runtime import upsert_tracked_detected

GRID = (32, 32)


def _seed_graph():
    feats = {"buildings": [{"name": "bldg_0", "cells": [[r, c] for r in range(5, 9) for c in range(5, 9)],
                            "centroid_row": 7, "centroid_col": 7}]}
    return build_object_graph(feats, cell_size_m=2.0)


def _node(class_name: str = "car") -> MapObject:
    return MapObject(
        id="detected_pending", class_name=class_name, name=None, tags={},
        centroid=(0, 0), bbox=(0, 0, 0, 0), cell_count=1,
        orientation=None, orientation_confidence=0.0, provenance="detected",
        confidence=0.8, first_seen_t=0.0, last_seen_t=0.0, observation_count=0,
    )


def _square_mask(r0, c0, side):
    m = np.zeros(GRID, dtype=bool)
    m[r0:r0 + side, c0:c0 + side] = True
    return m


def _cells_of(mask):
    rs, cs = np.where(mask)
    return {(int(r), int(c)) for r, c in zip(rs.tolist(), cs.tolist())}


def _sparse(mask):
    """The projector's sparse output shape for a dense fixture mask."""
    rs, cs = np.where(mask)
    return {"rows": rs.astype(np.int32), "cols": cs.astype(np.int32)}


def _upsert(g, track_id, mask, *, class_name="car", now=0.0):
    return upsert_tracked_detected(g, track_id=track_id, base_node=_node(class_name), **_sparse(mask),
                                   confidence=0.8, now=now)


def test_fresh_detection_stores_cells_matching_projection():
    g = _seed_graph()
    mask = _square_mask(10, 12, 3)
    node = g.nodes[_upsert(g, "t1", mask)]
    assert node.cells.dtype == np.int32
    assert node.cells.shape == (9, 2)
    assert _cells_of(mask) == {(int(r), int(c)) for r, c in node.cells}
    assert node.cell_count == 9


def test_repeat_sighting_restamps_cells_and_keeps_one_node():
    g = _seed_graph()
    fid0 = _upsert(g, "t1", _square_mask(10, 12, 3))
    moved = _square_mask(14, 16, 4)
    assert _upsert(g, "t1", moved, now=1.0) == fid0
    node = g.nodes[fid0]
    assert node.observation_count == 2
    assert _cells_of(moved) == {(int(r), int(c)) for r, c in node.cells}


def test_car_and_gray_car_stay_separate_layers():
    """Identical footprints under two track ids stay two nodes, each keeping its class."""
    g = _seed_graph()
    mask = _square_mask(10, 10, 6)
    fid_car = _upsert(g, "t1", mask)
    fid_gray = _upsert(g, "t2", mask, class_name="gray car", now=1.0)
    assert fid_car != fid_gray
    assert g.nodes[fid_car].class_name == "car"
    assert g.nodes[fid_gray].class_name == "gray car"
    assert sorted(n.class_name for n in g.nodes.values() if n.provenance == "detected") == ["car", "gray car"]


def test_a_repeat_with_no_cells_returns_its_own_id():
    g = _seed_graph()

    def upsert(tid, rows, cols):
        return upsert_tracked_detected(g, track_id=tid, base_node=_node(), rows=np.array(rows, dtype=np.int32),
                                       cols=np.array(cols, dtype=np.int32), confidence=0.9, now=1.0)

    a = upsert("wt_a", [5, 5], [5, 6])
    b = upsert("wt_b", [9, 9], [9, 10])
    assert upsert("wt_a", [], []) == a
    assert upsert("wt_b", [10], [10]) == b
