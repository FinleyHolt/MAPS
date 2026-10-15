"""Runtime graph mutation: a detection folds onto the ``detected_<i>`` node
carrying its exact ``track_id`` or mints one; a prior gains perceived colour."""

from __future__ import annotations

import numpy as np

from maps.sws.graph.builder import _mask_centerline
from maps.sws.graph.model import MapObject, ObjectGraph


# The color attrs of one best-view read, overwritten together.
_BEST_VIEW_COLOR_KEYS = ("color", "color_conf", "color_shade", "color_flag", "color_rgb")


def apply_best_view_color(node, color_attrs: dict, px: int) -> bool:
    """Best-view color, the same for detected and prior nodes.

    Replace the node's color buckets with ``color_attrs`` iff this frame's
    in-view mask pixel count ``px`` exceeds the stored ``best_view_px``, so the
    color comes from the frame where the object subtended the most pixels. All
    color keys move together; an empty value clears its key. True when it
    replaced.

    Callers pass ``px`` > 0 only with a real color read: a larger view clears
    every key absent from ``color_attrs``."""
    try:
        px = int(px or 0)
    except (TypeError, ValueError):
        px = 0
    if px <= 0 or px <= node.best_view_px:
        return False
    node.best_view_px = px
    for key in _BEST_VIEW_COLOR_KEYS:
        val = color_attrs.get(key)
        val_s = "" if val is None else str(val).strip()
        if val_s:
            node.semantic_attributes[key] = {val_s: 1}
        else:
            node.semantic_attributes.pop(key, None)
    return True


def upsert_prior_percept(graph: ObjectGraph, fid: str, attrs: dict, *, px: int = 0) -> bool:
    """Merge a perceived attribute set (the colour read over the prior's
    footprint) onto a prebake node: colour keys follow
    :func:`apply_best_view_color` on ``px``, any other key accumulates as
    ``{value: count}``. True when enriched (a non-empty colour read counts
    even when it lost the best view); a missing fid, a non-prebake node or
    empty ``attrs`` returns False."""
    if not attrs:
        return False
    node = graph.nodes.get(str(fid))
    if node is None or node.provenance != "prebaked":
        return False
    color_attrs = {k: attrs[k] for k in attrs if k in _BEST_VIEW_COLOR_KEYS}
    other_attrs = {k: attrs[k] for k in attrs if k not in _BEST_VIEW_COLOR_KEYS}
    touched = apply_best_view_color(node, color_attrs, px)
    # Any non-empty color read is a sighting, even one that lost the best view.
    if color_attrs and any(str(v).strip() for v in color_attrs.values()):
        touched = True
    for key, value in other_attrs.items():
        key_s = str(key).strip()
        val_s = str(value).strip()
        if not key_s or not val_s:
            continue
        bucket = node.semantic_attributes.setdefault(key_s, {})
        bucket[val_s] = bucket.get(val_s, 0) + 1
        touched = True
    return touched


def _next_detected_id(graph: ObjectGraph) -> str:
    """The next ``detected_<i>`` id: this module mints them from 0 and nothing
    removes a node, so the count of detected ids is the next index."""
    return f"detected_{sum(fid.startswith('detected_') for fid in graph.nodes)}"


def upsert_detected_centroid_only(graph: ObjectGraph, node: MapObject) -> str:
    """Mint a node for a detection without a ``track_id``; its new id."""
    if node.provenance != "detected":
        raise ValueError(
            "upsert_detected_centroid_only only accepts provenance='detected' nodes"
        )
    fid = _next_detected_id(graph)
    graph.nodes[fid] = node
    return fid


def _cells_from_mask(hm_rows: np.ndarray, hm_cols: np.ndarray) -> np.ndarray:
    """The ``(N, 2) int32`` (row, col) footprint of a projected detection, the
    format prebake nodes carry."""
    return np.stack((hm_rows, hm_cols), axis=1).astype(np.int32)


def _stamp_cells(
    node: MapObject, hm_rows: np.ndarray, hm_cols: np.ndarray, cell_size_m: float
) -> None:
    """Set a detected node's ``cells`` and, if it has none yet, the bake's own
    centerline family (``builder._mask_centerline``) derived from them.

    ponytail: derive-if-absent, so the centerline stays the first frame's
    while ``cells`` are re-stamped every frame; re-deriving changes the geom
    surface for detected fids and needs a paired SR read.
    """
    node.cells = _cells_from_mask(hm_rows, hm_cols)
    if node.centerline is not None or len(node.cells) < 2:
        return
    geom = _mask_centerline(node.cells, cell_size_m=float(cell_size_m))
    if geom is not None:
        node.centerline, node.axis_length_m = geom


def upsert_tracked_detected(
    graph: ObjectGraph,
    *,
    track_id: str,
    base_node: MapObject,
    rows: np.ndarray,
    cols: np.ndarray,
) -> str:
    """Upsert a tracked detection from its footprint cells on the tile grid
    (``rows``/``cols``, row-major). A first sighting of ``track_id`` inserts
    ``base_node`` with its geometry set from the cells; a repeat updates the
    node carrying that ``track_id`` in place. Returns the fid."""
    if base_node.provenance != "detected":
        raise ValueError(
            "upsert_tracked_detected only accepts provenance='detected' nodes"
        )

    hm_rows = np.asarray(rows)
    hm_cols = np.asarray(cols)

    existing_fid = _find_track(graph, track_id)

    if hm_rows.size == 0:
        return existing_fid or ""

    centroid_rc = (int(round(float(hm_rows.mean()))), int(round(float(hm_cols.mean()))))

    if existing_fid is None:
        fid = _next_detected_id(graph)
        base_node.track_id = track_id
        base_node.centroid = centroid_rc
        _stamp_cells(base_node, hm_rows, hm_cols, graph.cell_size_m)
        graph.nodes[fid] = base_node
        return fid

    # Repeat sighting: the latest observation's footprint, contour and xy win.
    node = graph.nodes[existing_fid]
    node.centroid = centroid_rc
    _stamp_cells(node, hm_rows, hm_cols, graph.cell_size_m)
    # A contour-less repeat keeps the stored contour.
    incoming_contour = base_node.world_contour
    if incoming_contour:
        node.world_contour = incoming_contour
    incoming_wxy = base_node.world_xy
    if incoming_wxy is not None:
        node.world_xy = (float(incoming_wxy[0]), float(incoming_wxy[1]))
    # base_node carries this frame's mask pixel count and color read, one
    # value per bucket.
    apply_best_view_color(
        node, {k: next(iter(v)) for k, v in base_node.semantic_attributes.items() if v}, base_node.best_view_px
    )
    return existing_fid


def _find_track(graph: ObjectGraph, track_id: str) -> str | None:
    """The detected node carrying ``track_id`` (the whole re-id rule), else None."""
    for fid, node in graph.nodes.items():
        if node.provenance == "detected" and node.track_id == track_id:
            return fid
    return None
