"""Parity tests for batched tracked-replace vs sequential single-replace.

The batched path must produce the same final node/ledger state as looping
``upsert_tracked_detected`` one track at a time.
"""

from __future__ import annotations

import numpy as np

from maps.sws.graph import (
    attach_mask_cache,
    upsert_tracked_detected,
    upsert_tracked_detected_batch,
)
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
    attach_mask_cache(graph, grid, features=features)
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


def _random_mask(rng, grid, cells: int):
    H, W = grid
    side = max(3, int(np.ceil(np.sqrt(cells * 1.5))))
    side = min(side, min(H, W) - 1)
    r0 = int(rng.integers(0, H - side))
    c0 = int(rng.integers(0, W - side))
    mask = np.zeros(grid, dtype=bool)
    mask[r0 : r0 + side, c0 : c0 + side] = True
    return mask


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


def test_car_and_gray_car_stay_separate_layers_single_path(monkeypatch):
    """Two-layer contract (tracked single path): the same physical car
    detected as a ``car`` mask (track t1) AND an identically-projected
    ``gray car`` mask (track t2) must NOT fold. Re-id is exact ``track_id``
    only, so the identical footprints stay two nodes and each keeps its own
    class — the agent counts the bare-class layer and reads color off the
    overlapping one. Prebake attribution is disabled so the detections mint
    discrete ``detected_<N>`` nodes (it is an orthogonal detected→prebake
    lane)."""
    monkeypatch.setenv("WORLD_STATE_DISABLE_PREBAKE_ATTRIBUTION", "1")
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
        frame_idx=0,
        now=0.0,
    )
    fid_gray = upsert_tracked_detected(
        graph,
        track_id="t2",
        base_node=_make_classed_tracked_node("t2", "gray car", cells, 0.8),
        **_sparse(mask),
        confidence=0.8,
        frame_idx=1,
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


def test_car_and_gray_car_stay_separate_layers_batch_path(monkeypatch):
    """Same two-layer contract through ``upsert_tracked_detected_batch`` —
    both masks delivered in one batched call still resolve to two distinct
    nodes."""
    monkeypatch.setenv("WORLD_STATE_DISABLE_PREBAKE_ATTRIBUTION", "1")
    grid = (32, 32)
    graph = _make_seed_graph(grid=grid)
    mask = _square_mask(grid, 10, 10, 6)
    cells = _mask_cells_list(mask)

    replaces = [
        {
            "track_id": "t1",
            "base_node": _make_classed_tracked_node("t1", "car", cells, 0.8),
            **_sparse(mask),
            "confidence": 0.8,
            "frame_idx": 0,
            "now": 0.0,
            "evidence_id": None,
        },
        {
            "track_id": "t2",
            "base_node": _make_classed_tracked_node("t2", "gray car", cells, 0.8),
            **_sparse(mask),
            "confidence": 0.8,
            "frame_idx": 1,
            "now": 1.0,
            "evidence_id": None,
        },
    ]
    out = upsert_tracked_detected_batch(graph, replaces, grid_shape=grid)

    assert len(out) == 2
    assert len(set(out)) == 2
    assert {graph.nodes[out[0]].class_name, graph.nodes[out[1]].class_name} == {
        "car",
        "gray car",
    }
    assert "car" in graph.objects_by_class and "gray car" in graph.objects_by_class
    counts = _detected_class_counts(graph)
    assert counts["car"] == 1 and counts["gray car"] == 1


def test_batch_with_only_fresh_tracks_falls_through_to_single():
    """Fresh tracks should be inserted via the single-path loop; the
    batched optimization is only for repeat-sights. No crash, each track
    gets a fid."""
    grid = (16, 16)
    g = _make_seed_graph(grid=grid)
    rng = np.random.default_rng(3)
    replaces = []
    for i in range(3):
        tid = f"fresh_{i}"
        m = _random_mask(rng, grid, cells=8)
        replaces.append(
            {
                "track_id": tid,
                "base_node": _make_tracked_node(tid, 0.8),
                **_sparse(m),
                "confidence": 0.8,
                "frame_idx": 0,
                "now": 0.0,
            }
        )
    fids = upsert_tracked_detected_batch(g, replaces)
    assert len(fids) == 3
    assert len(set(fids)) == 3
    assert all(f in g.nodes for f in fids)
