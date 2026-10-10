"""The graph's step index and clock survive a save/load and an upsert."""

from __future__ import annotations

from maps.sws import MapObject, ObjectGraph
from maps.sws.graph.io import load_graph, save_graph
from maps.sws.graph.runtime import upsert_detected_centroid_only


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
        step_index=7,
        tau=123.456,
    )
    save_graph(graph, str(tmp_path / "t"))
    restored = load_graph(str(tmp_path / "t"))
    assert restored.step_index == 7
    assert restored.tau == 123.456


def test_upsert_detected_centroid_only_sets_step_index():
    graph = ObjectGraph(nodes={}, cell_size_m=5.0)
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
    fid = upsert_detected_centroid_only(graph, det, now=100.0, step_index=3)
    assert graph.step_index == 3
    assert graph.tau == 100.0
    assert graph.nodes[fid].observation_count == 1
