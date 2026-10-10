"""MapObject runtime fields round-trip through save_graph / load_graph."""

from __future__ import annotations

from maps.sws.graph.io import load_graph, save_graph
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
    )
    return ObjectGraph(
        nodes={"buildings_0": prebaked, "detected_0": detected},
        cell_size_m=5.0,
    )


def _roundtrip(g: ObjectGraph, tmp_path) -> ObjectGraph:
    save_graph(g, str(tmp_path / "t"))
    return load_graph(str(tmp_path / "t"))


def test_roundtrip_preserves_runtime_fields(tmp_path):
    g = _graph_with_detected()
    g2 = _roundtrip(g, tmp_path)
    det = g2.nodes["detected_0"]
    assert det.provenance == "detected"
    assert det.first_seen_t == 100.0
    assert det.last_seen_t == 101.5
    assert det.observation_count == 3
    assert abs(det.confidence - 0.82) < 1e-9


def test_prebaked_default_provenance_preserved(tmp_path):
    g = _graph_with_detected()
    g2 = _roundtrip(g, tmp_path)
    pre = g2.nodes["buildings_0"]
    assert pre.provenance == "prebaked"
    assert pre.first_seen_t is None
    assert pre.observation_count == 0
    assert pre.confidence == 1.0
