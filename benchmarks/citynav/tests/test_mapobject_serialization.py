"""MapObject runtime fields round-trip through serialize / deserialize."""

from __future__ import annotations

from maps.sws.graph.io import deserialize_graph, serialize_graph
from maps.sws.graph.model import ObjectGraph, MapObject


def _graph_with_detected() -> ObjectGraph:
    prebaked = MapObject(
        id="buildings_0",
        class_name="buildings",
        name="Church",
        tags={},
        centroid=(10, 10),
        bbox=(9, 9, 11, 11),
        cell_count=9,
        orientation=None,
        orientation_confidence=0.0,
    )
    detected = MapObject(
        id="detected_0",
        class_name="red car",
        name="red car",
        tags={"label": "red car"},
        centroid=(12, 14),
        bbox=(12, 14, 12, 14),
        cell_count=1,
        orientation=None,
        orientation_confidence=0.0,
        provenance="detected",
        first_seen_t=100.0,
        last_seen_t=101.5,
        observation_count=3,
        confidence=0.82,
        embedding=[0.1, 0.2, 0.3],
    )
    return ObjectGraph(
        nodes={"buildings_0": prebaked, "detected_0": detected},
        cell_size_m=5.0,
    )


def test_roundtrip_preserves_runtime_fields():
    g = _graph_with_detected()
    blob = serialize_graph(g)
    g2 = deserialize_graph(blob)
    det = g2.nodes["detected_0"]
    assert det.provenance == "detected"
    assert det.first_seen_t == 100.0
    assert det.last_seen_t == 101.5
    assert det.observation_count == 3
    assert abs(det.confidence - 0.82) < 1e-9
    assert det.embedding == [0.1, 0.2, 0.3]


def test_prebaked_default_provenance_preserved():
    g = _graph_with_detected()
    g2 = deserialize_graph(serialize_graph(g))
    pre = g2.nodes["buildings_0"]
    assert pre.provenance == "prebaked"
    assert pre.first_seen_t is None
    assert pre.observation_count == 0
    assert pre.confidence == 1.0
