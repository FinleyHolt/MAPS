"""Runtime graph mutation: the track_id-keyed detected-node upsert.

A detection folds onto its ``detected_<N>`` node by exact ``track_id``
(``_find_track``; the world tracker's id is the sole re-id signal) or mints a
new one. A runtime node differs from a prebake node by
``provenance="detected"`` and its per-observation counters. The graph is
nodes-only; spatial relations are computed from node xy/cells by the
geom_compute ops. ``upsert_detected_centroid_only`` mints a node for a
detection without a track_id.
"""

from __future__ import annotations

import json
import time

import numpy as np

from maps.sws.graph.model import MapObject, ObjectGraph


# The color attrs of one best-view read, overwritten together.
_BEST_VIEW_COLOR_KEYS = ("color", "color_conf", "color_shade", "color_flag", "color_rgb")


def _flat_color(node) -> dict:
    """Top-count color value per key off a freshly-built node's {value:count} buckets."""
    sa = getattr(node, "semantic_attributes", None) or {}
    out: dict = {}
    for key in _BEST_VIEW_COLOR_KEYS:
        bucket = sa.get(key)
        if isinstance(bucket, dict) and bucket:
            out[key] = max(bucket.items(), key=lambda kv: kv[1])[0]
    return out


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
    if px <= 0 or px <= int(getattr(node, "best_view_px", 0) or 0):
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


def upsert_prior_percept(
    graph: ObjectGraph,
    fid: str,
    attrs: dict,
    *,
    px: int = 0,
    now: float | None = None,
) -> bool:
    """Merge a perceived attribute set onto a prebake node.

    ``attrs`` is the color read over the prior's projected footprint, the
    same extractor a detection gets (``perception.color``), e.g.
    ``{"color": "blue 0.7 / black 0.2", "color_conf": "0.83", ...}``. Color
    keys follow :func:`apply_best_view_color` on ``px``; any other key
    accumulates as ``{value: count}``. Every non-empty sighting bumps
    ``observation_count`` / ``first_seen_t`` / ``last_seen_t`` so the prior
    surfaces as an observed candidate. Only pixel-derived attributes land
    here; the node's geometry is untouched.

    True when the node was enriched. A missing fid, a non-prebake node or
    empty ``attrs`` returns False; it never raises.
    """
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict) or not attrs:
        return False
    node = nodes.get(str(fid))
    if node is None or getattr(node, "provenance", None) != "prebaked":
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
    if not touched:
        return False
    now = now if now is not None else time.time()
    if node.first_seen_t is None:
        node.first_seen_t = now
    node.last_seen_t = now
    node.observation_count += 1
    return True


def _next_detected_id(graph: ObjectGraph) -> str:
    """The next ``detected_<idx>`` id for ``graph``: a counter cached on the
    graph, seeded once from the highest existing id."""
    counter = graph.__dict__.get("_next_detected_idx")
    if counter is None:
        max_idx = -1
        for nid in graph.nodes:
            if nid.startswith("detected_"):
                try:
                    max_idx = max(max_idx, int(nid[len("detected_") :]))
                except ValueError:
                    continue
        counter = max_idx + 1
    fid = f"detected_{counter}"
    while fid in graph.nodes:
        counter += 1
        fid = f"detected_{counter}"
    graph.__dict__["_next_detected_idx"] = counter + 1
    return fid


def upsert_detected_centroid_only(
    graph: ObjectGraph,
    node: MapObject,
    *,
    now: float | None = None,
    step_index: int | None = None,
) -> str:
    """Mint a centroid-only detected node for a detection without a
    ``track_id``; the new ``detected_<N>`` id. Tracked detections go through
    :func:`upsert_tracked_detected`."""
    if node.provenance != "detected":
        raise ValueError(
            "upsert_detected_centroid_only only accepts provenance='detected' nodes"
        )
    now = now if now is not None else time.time()
    fid = _next_detected_id(graph)
    node.id = fid
    node.first_seen_t = now
    node.last_seen_t = now
    node.observation_count = 1
    graph.nodes[fid] = node
    if step_index is not None:
        graph.step_index = step_index
    graph.tau = now
    return fid


# The dedup record stamped on a detected node's ``tags['dedup']``: a first
# sighting, or a repeat re-identified by its exact track_id.
_DEDUP_NEW = json.dumps({"status": "new"})
_DEDUP_TRACK = json.dumps({"status": "confirmed_track", "primary": "track_id", "match_score": 1.0})


def _cells_from_mask(hm_rows: np.ndarray, hm_cols: np.ndarray) -> np.ndarray:
    """The ``(N, 2) int32`` (row, col) footprint of a projected detection, the
    format prebake nodes carry."""
    return np.stack((hm_rows, hm_cols), axis=1).astype(np.int32)


def _stamp_cells(
    node: MapObject, hm_rows: np.ndarray, hm_cols: np.ndarray, cell_size_m: float
) -> None:
    """Set a detected node's footprint and the centerline derived from it.

    The one place a detected node's ``cells`` are written. It calls the
    loader's own ``io.derive_centerlines``, so a detected node carries the
    centerline a loaded node would; the geom ops resolve a fid with a
    centerline as a line and one without as a point."""
    from maps.sws.graph.io import derive_centerlines

    node.cells = _cells_from_mask(hm_rows, hm_cols)
    derive_centerlines({node.id: node}, float(cell_size_m))


def upsert_tracked_detected(
    graph: ObjectGraph,
    *,
    track_id: str,
    base_node: MapObject,
    rows: np.ndarray,
    cols: np.ndarray,
    confidence: float,
    now: float | None = None,
) -> str:
    """Upsert a tracked detection from its projected ground cells.

    ``rows`` / ``cols`` are the deduplicated occupied cells of the
    detection's footprint on the tile grid, in row-major order
    (``TileProjectedMask.rows`` / ``.cols``).

    First sighting of ``track_id`` allocates a new ``detected_*`` node.

    Repeat sightings look up the existing node by ``track_id`` (ByteTrack
    is the sole dedup) and replace the node attrs in place.

    ``base_node`` carries the class/tags/label for a fresh insert; its
    centroid and footprint are set from the cells.
    """
    if base_node.provenance != "detected":
        raise ValueError(
            "upsert_tracked_detected only accepts provenance='detected' nodes"
        )

    now = now if now is not None else time.time()

    hm_rows = np.asarray(rows)
    hm_cols = np.asarray(cols)

    existing_fid = _find_track(graph, track_id)

    if hm_rows.size == 0:
        return existing_fid or ""

    cell_count = int(hm_rows.size)
    centroid_rc = (int(round(float(hm_rows.mean()))), int(round(float(hm_cols.mean()))))
    bbox = (
        int(hm_rows.min()),
        int(hm_cols.min()),
        int(hm_rows.max()),
        int(hm_cols.max()),
    )

    if existing_fid is None:
        fid = _next_detected_id(graph)
        base_node.id = fid
        base_node.track_id = track_id
        base_node.provenance = "detected"
        base_node.centroid = centroid_rc
        base_node.bbox = bbox
        base_node.cell_count = cell_count
        _stamp_cells(base_node, hm_rows, hm_cols, graph.cell_size_m)
        base_node.confidence = float(confidence)
        base_node.observation_count = 1
        base_node.first_seen_t = now
        base_node.last_seen_t = now
        base_node.tags["dedup"] = _DEDUP_NEW

        graph.nodes[fid] = base_node
        return fid

    # Repeat sighting: the latest observation's footprint, contour and xy win.
    node = graph.nodes[existing_fid]
    node.track_id = track_id
    node.centroid = centroid_rc
    node.bbox = bbox
    node.cell_count = cell_count
    _stamp_cells(node, hm_rows, hm_cols, graph.cell_size_m)
    # A contour-less repeat keeps the stored contour.
    incoming_contour = getattr(base_node, "world_contour", None)
    if incoming_contour:
        node.world_contour = incoming_contour
    incoming_wxy = getattr(base_node, "world_xy", None)
    if incoming_wxy is not None:
        node.world_xy = (float(incoming_wxy[0]), float(incoming_wxy[1]))
    # base_node carries this frame's mask pixel count and color read.
    apply_best_view_color(
        node, _flat_color(base_node), getattr(base_node, "best_view_px", 0)
    )
    node.observation_count += 1
    n_obs = node.observation_count
    node.confidence = (
        (node.confidence * (n_obs - 1) + float(confidence)) / n_obs
        if n_obs > 0
        else float(confidence)
    )
    node.last_seen_t = now
    node.tags["dedup"] = _DEDUP_TRACK

    return existing_fid


def _find_track(graph: ObjectGraph, track_id: str) -> str | None:
    """The detected node carrying ``track_id``, else None.

    The world tracker assigns one stable, class-gated ``track_id`` per
    physical object, so the exact id is the whole re-identification rule.
    """
    for fid, node in graph.nodes.items():
        if node.provenance == "detected" and node.track_id == track_id:
            return fid
    return None
