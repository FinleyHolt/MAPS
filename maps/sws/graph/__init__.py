"""Feature graph: spatial relationship model, I/O, and construction."""

from maps.sws.graph.io import (
    load_graph,
    save_graph,
)
from maps.sws.graph.mask_cache import (
    MaskCache,
    attach_mask_cache,
    build_mask_cache,
    get_mask_cache,
)
from maps.sws.graph.model import (
    ObjectGraph,
    MapObject,
)
from maps.sws.graph.runtime import (
    PrebakeEvidence,
    TrackContribution,
    get_prebake_evidence,
    revoke_track_contribution,
    upsert_detected_centroid_only,
    upsert_tracked_detected,
    upsert_tracked_detected_batch,
)

__all__ = [
    "MaskCache",
    "MapObject",
    "ObjectGraph",
    "PrebakeEvidence",
    "TrackContribution",
    "attach_mask_cache",
    "build_mask_cache",
    "build_object_graph",
    "get_mask_cache",
    "get_prebake_evidence",
    "load_graph",
    "revoke_track_contribution",
    "save_graph",
    "upsert_detected_centroid_only",
    "upsert_tracked_detected",
    "upsert_tracked_detected_batch",
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
