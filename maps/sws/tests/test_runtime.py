"""The builder's node."""

from __future__ import annotations


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
