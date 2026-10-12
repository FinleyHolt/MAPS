"""Graph serdes and the builder's node."""

from __future__ import annotations

import dataclasses

import numpy as np

from maps.sws.graph.model import MapObject, ObjectGraph
from maps.sws.graph.io import load_graph, save_graph


def _detected(centroid: tuple[int, int], **overrides) -> MapObject:
    node = MapObject(
        class_name="object",
        name=None,
        tags={},
        centroid=centroid,
        provenance="detected",
    )
    for key, value in overrides.items():
        setattr(node, key, value)
    return node


def test_save_load_round_trips_every_node_field(tmp_path):
    """A detection's runtime fields, counts (still ints), cells and contour, and
    a prebaked node's contour and defaults, all survive the nodes JSON."""
    det = _detected(
        (10, 12),
        semantic_attributes={"color": {"gray": 3, "white": 1}, "text_span": {"7B": 2}},
        cells=np.array([[10, 12], [10, 13]], dtype=np.int32),
        world_contour=[[30.5, 40.5], [31.0, 40.0], [31.5, 41.0]],
    )
    pre = MapObject(class_name="buildings", name="Church", tags={}, centroid=(10, 10),
                    world_contour=[[10.0, 20.0], [11.0, 20.0], [11.0, 21.0], [10.0, 20.0]])
    nodes = {"detected_0": det, "buildings_0": pre}
    save_graph(ObjectGraph(nodes=nodes, cell_size_m=5.0), str(tmp_path / "t"))
    restored = load_graph(str(tmp_path / "t"))
    for fid, node in nodes.items():
        got = restored.nodes[fid]
        for f in dataclasses.fields(MapObject):
            want, have = getattr(node, f.name), getattr(got, f.name)
            assert np.array_equal(want, have) if f.name == "cells" and want is not None else want == have, f.name
    assert all(type(c) is int for c in restored.nodes["detected_0"].semantic_attributes["color"].values())


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
