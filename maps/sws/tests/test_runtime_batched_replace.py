"""The tracked upsert: one node per track_id, whatever the footprint."""

from __future__ import annotations

import numpy as np

from maps.sws.graph import upsert_tracked_detected
from maps.sws.graph.builder import build_object_graph


def _make_features(n_prebake: int = 4, grid=(32, 32)) -> dict:
    """Build a small features dict: one building class, N blob features."""
    H, W = grid
    feats = []
    rng = np.random.default_rng(11)
    for i in range(n_prebake):
        r0 = int(rng.integers(2, H - 6))
        c0 = int(rng.integers(2, W - 6))
        cells = []
        for r in range(r0, r0 + 4):
            for c in range(c0, c0 + 4):
                cells.append([r, c])
        feats.append(
            {
                "name": f"bldg_{i}",
                "cells": cells,
                "centroid_row": r0 + 2,
                "centroid_col": c0 + 2,
            }
        )
    return {"buildings": feats}


def _make_seed_graph(grid=(32, 32)):
    features = _make_features(n_prebake=4, grid=grid)
    graph = build_object_graph(features, cell_size_m=5.0)
    return graph


def _make_tracked_node(track_id: str, confidence: float):
    from maps.sws.graph.model import MapObject

    return MapObject(
        id="detected_pending",
        class_name="buildings",
        name=f"track_{track_id}",
        tags={},
        centroid=(0, 0),
        bbox=(0, 0, 0, 0),
        cell_count=1,
        orientation=None,
        orientation_confidence=0.0,
        provenance="detected",
        confidence=float(confidence),
        first_seen_t=0.0,
        last_seen_t=0.0,
        observation_count=0,
    )


def _mask_cells_list(mask):
    rs, cs = np.where(mask)
    return [[int(r), int(c)] for r, c in zip(rs.tolist(), cs.tolist())]


def _make_classed_tracked_node(track_id: str, class_name: str, cells_list, confidence: float):
    """Like ``_make_tracked_node`` but parametrises ``class_name`` and SETS
    ``MapObject.cells`` (np.ndarray (N,2) int32), so the two-layer tests below
    carry a real footprint on each node — the base fixture hardcodes
    ``"buildings"`` and leaves ``.cells`` None."""
    from maps.sws.graph.model import MapObject

    cells_arr = np.asarray(cells_list, dtype=np.int32).reshape(-1, 2)
    return MapObject(
        id="detected_pending",
        class_name=class_name,
        name=f"track_{track_id}",
        tags={},
        centroid=(int(cells_arr[:, 0].mean()), int(cells_arr[:, 1].mean())),
        bbox=(
            int(cells_arr[:, 0].min()),
            int(cells_arr[:, 1].min()),
            int(cells_arr[:, 0].max()),
            int(cells_arr[:, 1].max()),
        ),
        cell_count=int(cells_arr.shape[0]),
        orientation=None,
        orientation_confidence=0.0,
        provenance="detected",
        confidence=float(confidence),
        first_seen_t=0.0,
        last_seen_t=0.0,
        observation_count=0,
        cells=cells_arr,
    )


def _sparse(mask):
    """The projector's sparse output shape for a dense fixture mask."""
    rs, cs = np.where(mask)
    return {"rows": rs.astype(np.int32), "cols": cs.astype(np.int32)}


def _square_mask(grid, r0, c0, side):
    mask = np.zeros(grid, dtype=bool)
    mask[r0 : r0 + side, c0 : c0 + side] = True
    return mask


def _detected_class_counts(graph):
    import collections

    counts: collections.Counter = collections.Counter()
    for n in graph.nodes.values():
        if n.provenance == "detected":
            counts[n.class_name] += 1
    return counts


def test_car_and_gray_car_stay_separate_layers_single_path():
    """Two-layer contract (tracked single path): the same physical car
    detected as a ``car`` mask (track t1) AND an identically-projected
    ``gray car`` mask (track t2) must NOT fold. Re-id is exact ``track_id``
    only, so the identical footprints stay two nodes and each keeps its own
    class — the agent counts the bare-class layer and reads color off the
    overlapping one."""
    grid = (32, 32)
    graph = _make_seed_graph(grid=grid)
    mask = _square_mask(grid, 10, 10, 6)  # identical footprints, distinct ids
    cells = _mask_cells_list(mask)

    fid_car = upsert_tracked_detected(
        graph,
        track_id="t1",
        base_node=_make_classed_tracked_node("t1", "car", cells, 0.8),
        **_sparse(mask),
        confidence=0.8,
        now=0.0,
    )
    fid_gray = upsert_tracked_detected(
        graph,
        track_id="t2",
        base_node=_make_classed_tracked_node("t2", "gray car", cells, 0.8),
        **_sparse(mask),
        confidence=0.8,
        now=1.0,
    )

    assert fid_car != fid_gray
    assert graph.nodes[fid_car].class_name == "car"
    assert graph.nodes[fid_gray].class_name == "gray car"
    assert "car" in graph.objects_by_class
    assert "gray car" in graph.objects_by_class
    counts = _detected_class_counts(graph)
    assert counts["car"] == 1
    assert counts["gray car"] == 1


def test_a_repeat_with_no_cells_returns_its_own_id():
    """A repeat sighting with an empty footprint returns its track's node id."""
    grid = (32, 32)
    g = _make_seed_graph(grid=grid)

    def rep(tid, rows, cols):
        return {
            "track_id": tid,
            "base_node": _make_tracked_node(tid, 0.9),
            "rows": np.array(rows, dtype=np.int32),
            "cols": np.array(cols, dtype=np.int32),
            "confidence": 0.9,
            "now": 1.0,
        }

    a = upsert_tracked_detected(g, **rep("wt_a", [5, 5], [5, 6]))
    b = upsert_tracked_detected(g, **rep("wt_b", [9, 9], [9, 10]))
    assert upsert_tracked_detected(g, **rep("wt_a", [], [])) == a
    assert upsert_tracked_detected(g, **rep("wt_b", [10], [10])) == b
