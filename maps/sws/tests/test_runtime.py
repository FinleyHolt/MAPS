"""Runtime graph serdes.

- ``save_graph`` / ``load_graph`` round-trip ``semantic_attributes`` through
  the nodes JSON, which is where str/int coercion at the wire layer would show.
- ``save_graph`` / ``load_graph`` preserve ``cells``, ``tags['height_m']`` and
  ``world_contour``.
"""

from __future__ import annotations

from maps.sws import MapObject, ObjectGraph
from maps.sws.graph.io import load_graph, save_graph


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


def test_semantic_attributes_round_trip_via_json(tmp_path):
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
    )
    save_graph(graph, str(tmp_path / "t"))
    restored = load_graph(str(tmp_path / "t"))
    out = restored.nodes["detected_0"].semantic_attributes
    assert out == {"color": {"gray": 3, "white": 1}, "text_span": {"7B": 2}}
    # Counts must remain ints (not str), keys must remain str.
    for key, counts in out.items():
        assert isinstance(key, str)
        for value, count in counts.items():
            assert isinstance(value, str)
            assert isinstance(count, int)


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
    fine footprint polygon). A serdes that drops the field drops it on every
    node, priors AND detections, while everything else survives. Pin both
    sides of the serdes so that regression cannot pass silently.
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
    )
    with tempfile.TemporaryDirectory() as d:
        stem = str(pathlib.Path(d) / "tile")
        save_graph(g, stem)
        g2 = load_graph(stem)

    assert g2.nodes["building_0"].world_contour == prior_contour
    assert g2.nodes["detected_0"].world_contour == det_contour
