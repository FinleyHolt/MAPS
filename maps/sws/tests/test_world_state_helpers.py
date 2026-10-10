"""Unit tests for WorldState properties and graph round-trip fields."""

from __future__ import annotations

from maps.sws import (
    GridDomain,
    MapObject,
    ObjectGraph,
    WorldState,
)
from maps.sws.graph.io import load_graph, save_graph
from maps.sws.graph.runtime import upsert_detected_centroid_only


def _make_ws(h: int = 4, w: int = 4) -> WorldState:
    return WorldState(
        grid_domain=GridDomain(height=h, width=w, cell_size_m=5.0),
    )


def test_classes_empty_when_graph_missing():
    ws = _make_ws()
    assert ws.classes == frozenset()


def test_classes_from_object_graph():
    node = MapObject(
        id="roads_0",
        class_name="roads",
        name=None,
        tags={},
        centroid=(1, 1),
        bbox=(1, 1, 1, 1),
        cell_count=1,
        orientation=None,
    )
    graph = ObjectGraph(
        nodes={"roads_0": node},
        cell_size_m=5.0,
        objects_by_class={"roads": {"roads_0": node}},
    )
    ws = _make_ws()
    ws.object_graph = graph
    assert "roads" in ws.classes


def test_objectgraph_step_index_round_trip(tmp_path):
    node = MapObject(
        id="roads_0",
        class_name="roads",
        name=None,
        tags={},
        centroid=(0, 0),
        bbox=(0, 0, 0, 0),
        cell_count=1,
        orientation=None,
    )
    graph = ObjectGraph(
        nodes={"roads_0": node},
        cell_size_m=5.0,
        objects_by_class={"roads": {"roads_0": node}},
        step_index=7,
        tau=123.456,
    )
    save_graph(graph, str(tmp_path / "t"))
    restored = load_graph(str(tmp_path / "t"))
    assert restored.step_index == 7
    assert restored.tau == 123.456


def test_upsert_detected_centroid_only_sets_step_index_and_evidence_refs():
    graph = ObjectGraph(nodes={}, cell_size_m=5.0, objects_by_class={})
    det = MapObject(
        id="detected_pending",
        class_name="object",
        name=None,
        tags={},
        centroid=(2, 2),
        bbox=(2, 2, 2, 2),
        cell_count=1,
        orientation=None,
        provenance="detected",
    )
    fid = upsert_detected_centroid_only(graph, det, now=100.0, step_index=3, evidence_id="evid-a")
    assert graph.step_index == 3
    assert graph.tau == 100.0
    assert graph.nodes[fid].evidence_refs == ["evid-a"]
