"""Runtime graph serdes.

- ``save_graph`` / ``load_graph`` round-trip a detection's runtime fields and
  ``semantic_attributes`` through the nodes JSON (where str/int coercion at the
  wire layer would show), and a prebaked node's defaults.
- ``save_graph`` / ``load_graph`` preserve ``world_contour``.
"""

from __future__ import annotations

from maps.sws.graph.model import MapObject, ObjectGraph
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


def test_runtime_fields_and_semantic_attributes_round_trip_via_json(tmp_path):
    node = _detected(
        (10, 12),
        semantic_attributes={"color": {"gray": 3, "white": 1}, "text_span": {"7B": 2}},
        first_seen_t=1.0,
        last_seen_t=4.0,
        observation_count=4,
        confidence=0.8,
    )
    prebaked = MapObject(id="buildings_0", class_name="buildings", name="Church", tags={},
                         centroid=(10, 10), bbox=(9, 9, 11, 11), cell_count=9, orientation=None)
    graph = ObjectGraph(
        nodes={"detected_0": node, "buildings_0": prebaked},
        cell_size_m=5.0,
    )
    save_graph(graph, str(tmp_path / "t"))
    restored = load_graph(str(tmp_path / "t"))
    det, pre = restored.nodes["detected_0"], restored.nodes["buildings_0"]
    assert (det.provenance, det.first_seen_t, det.last_seen_t, det.observation_count) == ("detected", 1.0, 4.0, 4)
    assert det.confidence == 0.8
    assert (pre.provenance, pre.first_seen_t, pre.observation_count, pre.confidence) == ("prebaked", None, 0, 1.0)
    out = det.semantic_attributes
    assert out == {"color": {"gray": 3, "white": 1}, "text_span": {"7B": 2}}
    # Counts must remain ints (not str), keys must remain str.
    for key, counts in out.items():
        assert isinstance(key, str)
        for value, count in counts.items():
            assert isinstance(value, str)
            assert isinstance(count, int)


def test_a_baked_feature_builds_its_node():
    """A bake row's cells, centroid and contour land on the node; its other
    keys become tags verbatim; its fid is ``<features-key>_<i>``."""
    from maps.sws.graph.builder import build_object_graph

    contour = [[20.0, 40.0], [22.0, 40.0], [22.0, 42.0]]
    feats = {"prior": [{"cells": [[10, 20], [10, 21], [11, 20], [11, 21]], "centroid_row": 10.6,
                        "centroid_col": 20.4, "world_contour": contour, "name": "Mill",
                        "orientation": "ne", "building": "yes", "source": "osm"}]}
    node = build_object_graph(feats, cell_size_m=2.0).nodes["prior_0"]
    assert (node.name, node.centroid, node.cells.shape, node.world_contour) == ("Mill", (11, 20), (4, 2), contour)
    assert node.tags == {"building": "yes", "source": "osm"}


def test_save_load_graph_preserves_world_contour_on_prior_and_detected():
    """``save_graph`` -> ``load_graph`` MUST preserve ``world_contour``.

    The SAM mask-tracking substrate rides on ``MapObject.world_contour`` (the
    fine footprint polygon). A serdes that drops the field drops it on every
    node, priors AND detections, while everything else survives. Pin both
    sides of the serdes so that regression cannot pass silently.
    """
    import pathlib
    import tempfile


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
