"""Runtime graph: re-identification, serdes, and view-frame geometry.

- ``_find_track`` re-identifies by exact ByteTrack ``track_id`` and nothing
  else; a different id mints a distinct node even on an identical footprint.
- ``serialize_graph`` / ``deserialize_graph`` round-trip ``semantic_attributes``
  through JSON, which is where str/int coercion at the wire layer would show.
- ``save_graph`` / ``load_graph`` preserve ``cells``, ``tags['height_m']`` and
  ``world_contour`` -- the last one against a measured 2026-07-04 loss.
- ``compute_view_frames`` puts each grid quadrant in the matching body-frame
  quadrant for a north-facing drone.
"""

from __future__ import annotations

import json
import math


from maps.sws import MapObject, ObjectGraph
from maps.sws.geometry import compute_view_frames
from maps.sws.graph.io import deserialize_graph, serialize_graph
from maps.sws.graph.runtime import _find_track


def _detected(centroid: tuple[int, int], **overrides) -> MapObject:
    node = MapObject(
        id="detected_pending",
        class_name="object",
        name=None,
        tags={},
        centroid=centroid,
        bbox=(centroid[0], centroid[1], centroid[0], centroid[1]),
        cell_count=1,
        orientation=None,
        provenance="detected",
    )
    for key, value in overrides.items():
        setattr(node, key, value)
    return node


def test_semantic_attributes_round_trip_via_json():
    node = _detected(
        (10, 12),
        semantic_attributes={"color": {"gray": 3, "white": 1}, "text_span": {"7B": 2}},
        first_seen_t=1.0,
        last_seen_t=4.0,
        observation_count=4,
        confidence=0.8,
    )
    graph = ObjectGraph(
        nodes={"detected_0": node},
        cell_size_m=5.0,
        objects_by_class={"object": {"detected_0": node}},
    )
    data = serialize_graph(graph)
    # Round-trip through JSON to flush str/int coercion at the wire layer.
    restored = deserialize_graph(json.loads(json.dumps(data)))
    out = restored.nodes["detected_0"].semantic_attributes
    assert out == {"color": {"gray": 3, "white": 1}, "text_span": {"7B": 2}}
    # Counts must remain ints (not str), keys must remain str.
    for key, counts in out.items():
        assert isinstance(key, str)
        for value, count in counts.items():
            assert isinstance(value, str)
            assert isinstance(count, int)


def test_compute_view_frames_quadrants_against_north_facing_drone():
    # 10x10 grid, 5 m cells, tile origin at world (0, 0). The drone sits at
    # the tile's center facing north, so a node placed in each grid quadrant
    # should land in the matching body-frame quadrant.
    H, W = 10, 10
    cell = 5.0
    drone_x = (W / 2) * cell  # 25 m east of x_min
    drone_y = (H / 2) * cell  # 25 m north of y_min

    # Row 0 = south = y_min, row H-1 = north = y_max (bake convention).
    # row 7, col 2 — north-west of drone (forward, left)
    # row 7, col 7 — north-east of drone (forward, right)
    # row 2, col 2 — south-west of drone (back, left)
    # row 2, col 7 — south-east of drone (back, right)
    centroids = [(7, 2), (7, 7), (2, 2), (2, 7)]
    frames = compute_view_frames(
        centroids,
        drone_x_m=drone_x,
        drone_y_m=drone_y,
        yaw_deg=0.0,
        tile_origin_xy=(0.0, 0.0),
        cell_size_m=cell,
        grid_height=H,
    )
    quadrants = [vf.quadrant for vf in frames]
    assert quadrants == ["front-left", "front-right", "back-left", "back-right"]
    # Range is symmetric across the four corners.
    ranges = [vf.range_m for vf in frames]
    assert max(ranges) - min(ranges) < 1e-3


def test_find_track_exact_track_id_only():
    """``_find_track`` re-identifies ONLY by exact ``track_id`` — ByteTrack
    is the sole dedup, with no geometry fallback. A query for the node's own
    track_id returns it; any other track_id returns None even when the
    existing node's class/footprint would have matched under a geometry
    path, so a distinct node would be minted."""
    car = MapObject(
        id="detected_0",
        class_name="car",
        name="car",
        tags={},
        centroid=(10, 10),
        bbox=(10, 10, 11, 11),
        cell_count=4,
        orientation=None,
        provenance="detected",
        track_id="wt_existing",
    )
    graph = ObjectGraph(
        nodes={"detected_0": car}, cell_size_m=2.0, objects_by_class={}
    )
    # Same track_id -> re-identifies the existing node.
    assert _find_track(graph, "wt_existing") == "detected_0"

    # Different track_id -> None even though the existing node's class and
    # footprint match (no geometry fallback; the ByteTrack track_id is the
    # only key).
    assert _find_track(graph, "wt_new") is None


# ---------------------------------------------------------------------------
# Prebake nodes carry ``cells`` so the visibility-driven pose resolver
# (``tour_synth._candidate_target_cells``) can read a real target mask off
# the runtime graph. Detected nodes already carry ``cells`` via
# ``_detection_to_mapobject``; prebake nodes used to leave it ``None``,
# which silently broke every side-view resolution against a prebake fid.
# ---------------------------------------------------------------------------


def test_build_object_graph_populates_cells_and_height_tag():
    """``_build_node`` stamps ``cells`` + ``tags['height_m']`` from features."""
    import pathlib
    import tempfile

    from maps.sws.graph import load_graph, save_graph
    from maps.sws.graph.builder import build_object_graph

    features = {
        "building": [
            {
                "id": "ignored_when_cityref_id_present",
                "kind": "building",
                "cells": [[10, 20], [10, 21], [11, 20], [11, 21]],
                "centroid_row": 10.5,
                "centroid_col": 20.5,
                "height_m": 9.19,
                "altitude_m": 18.485,
                "cityref_id": "38",
                "object_type": "Building",
            }
        ],
    }
    g = build_object_graph(features, cell_size_m=2.0)
    # CityRefer features are keyed by ``cityref_<id>``; the builder's fid
    # scheme is the runtime contract.
    assert "cityref_38" in g.nodes
    node = g.nodes["cityref_38"]
    assert node.tags["height_m"] == "9.190"
    assert node.cells is not None
    assert node.cells.shape == (4, 2)

    # Round-trip via save_graph / load_graph pins the io serdes.
    with tempfile.TemporaryDirectory() as d:
        stem = str(pathlib.Path(d) / "tile")
        save_graph(g, stem)
        g2 = load_graph(stem)
    n2 = g2.nodes["cityref_38"]
    assert n2.tags["height_m"] == "9.190"
    assert n2.cells is not None
    assert n2.cells.shape == (4, 2)


def test_save_load_graph_preserves_world_contour_on_prior_and_detected():
    """``save_graph`` -> ``load_graph`` MUST preserve ``world_contour``.

    The SAM mask-tracking substrate rides on ``MapObject.world_contour`` (the
    fine footprint polygon). The world_state ingest server hands the runtime a
    graph over exactly this ``save_graph`` -> bytes -> ``load_graph`` round-trip
    (``world_state_ingest_server._snapshot_bundle`` /
    ``world_state_ingest_client._load_snapshot``). A stale server whose io
    serdes predates this field silently DROPPED it on every node — priors AND
    detections — which read as "world_contour_coverage 0/32 + 0/242" on the
    2026-07-04 real-stack smoke while color (long-serialized) survived. Pin
    both sides of the serdes so that regression cannot pass silently.
    """
    import pathlib
    import tempfile

    from maps.sws.graph import load_graph, save_graph

    prior_contour = [[10.0, 20.0], [11.0, 20.0], [11.0, 21.0], [10.0, 21.0]]
    det_contour = [[30.5, 40.5], [31.0, 40.0], [31.5, 41.0]]
    prior = MapObject(
        id="building_0",
        class_name="building",
        name="Church",
        tags={},
        centroid=(10, 10),
        bbox=(8, 8, 12, 12),
        cell_count=25,
        orientation=None,
        orientation_confidence=0.0,
        provenance="prebaked",
        world_contour=prior_contour,
    )
    detected = MapObject(
        id="detected_0",
        class_name="car",
        name="car",
        tags={"label": "car"},
        centroid=(30, 40),
        bbox=(30, 40, 31, 41),
        cell_count=3,
        orientation=None,
        orientation_confidence=0.0,
        provenance="detected",
        confidence=0.9,
        world_contour=det_contour,
    )
    g = ObjectGraph(
        nodes={"building_0": prior, "detected_0": detected},
        cell_size_m=2.0,
        objects_by_class={
            "building": {"building_0": prior},
            "car": {"detected_0": detected},
        },
    )
    with tempfile.TemporaryDirectory() as d:
        stem = str(pathlib.Path(d) / "tile")
        save_graph(g, stem)
        g2 = load_graph(stem)

    assert g2.nodes["building_0"].world_contour == prior_contour
    assert g2.nodes["detected_0"].world_contour == det_contour
