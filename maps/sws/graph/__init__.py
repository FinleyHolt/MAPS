"""Feature graph: spatial relationship model, I/O, and construction."""

from maps.sws.graph.io import (
    load_graph,
    save_graph,
)
from maps.sws.graph.model import (
    ObjectGraph,
    MapObject,
)
from maps.sws.graph.runtime import (
    upsert_detected_centroid_only,
    upsert_tracked_detected,
)

__all__ = [
    "MapObject",
    "ObjectGraph",
    "build_object_graph",
    "load_graph",
    "save_graph",
    "upsert_detected_centroid_only",
    "upsert_tracked_detected",
]


def build_object_graph(
    features: dict,
    *,
    cell_size_m: float,
) -> ObjectGraph:
    """Build the object graph (nodes only) from features.json data.

    Lazy import so package load stays cheap.
    """
    from maps.sws.graph.builder import build_object_graph as _build

    return _build(features, cell_size_m=cell_size_m)
