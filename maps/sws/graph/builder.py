"""Feature graph construction from tile features.

Builds ``MapObject`` nodes (centroid, bbox, cells, orientation, centerline)
from ``features.json`` data. No edges are built or stored — spatial relations
are computed geometrically from node xy/cells by the geom_compute ops at
query time.
"""

from __future__ import annotations

import logging

import numpy as np
from maps.sws.spatial import compute_orientation as _compute_orientation

from maps.sws.graph.model import (
    ObjectGraph,
    MapObject,
)

# Keys in the ``features`` dict that carry non-feature metadata rather than a
# feature list. ``build_object_graph`` skips them.
_NON_FEATURE_KEYS: frozenset[str] = frozenset(
    {"bbox", "georef", "_tile_name", "nlcd_zones"}
)


# Feature keys stored on dedicated MapObject fields or consumed by the builder,
# never as tags; everything else rides into ``MapObject.tags`` verbatim.
# ``descriptions`` is the GT-label wall and never reaches a node.
_TAG_SKIP_KEYS = frozenset(
    {"cells", "world_contour", "centroid_row", "centroid_col", "orientation", "name", "descriptions"}
)


def _extract_tags(feat: dict) -> dict[str, str]:
    """Every non-geometry, non-bookkeeping feature key becomes a tag,
    verbatim. String values pass through; string lists join comma-sorted."""
    tags: dict[str, str] = {}
    for key, val in feat.items():
        if val is None or key in _TAG_SKIP_KEYS:
            continue
        if isinstance(val, str):
            tags[key] = val
        elif isinstance(val, list):
            unique = sorted({v for v in val if isinstance(v, str) and v})
            if unique:
                tags[key] = ",".join(unique)
    return tags


def _compute_bbox(cells: list[list[int]]) -> tuple[int, int, int, int]:
    """Compute bounding box from cell list."""
    arr = np.array(cells)
    return (
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 0].max()),
        int(arr[:, 1].max()),
    )


def _mask_centerline(
    arr: np.ndarray, *, cell_size_m: float
) -> tuple[list[list[float]], list[float], float, float, str] | None:
    """Raw, parameter-free principal-axis centerline of a mask.

    Returns ``(centerline_rc, arc_m, length_m, residual_m, method)`` or
    ``None`` when the mask is too small / degenerate to admit an axis.

    The centerline is the mask's medial line along its PCA principal axis:
    every cell is projected onto the axis, binned by the integer projection
    step (the grid's own resolution), and each bin contributes its mean cell
    position. Vertices are in the ``(row, col)`` grid frame; the meter scalars
    are frame-invariant.

    ``residual_m`` is the max perpendicular distance (meters) of any mask cell
    to the straight PCA line: about half the width for a straight mask, more
    for a curved one.
    """
    if arr is None or len(arr) < 2:
        return None
    pts = np.asarray(arr, dtype=float)
    mean = pts.mean(axis=0)
    centered = pts - mean
    try:
        cov = np.cov(centered.T)
        _evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    # eigh returns ascending eigenvalues: column 1 is the major (principal) axis.
    e1 = evecs[:, 1]
    e2 = evecs[:, 0]
    s = centered @ e1  # signed projection along the principal axis (cells)
    lat = centered @ e2  # signed lateral offset (cells)
    residual_m = float(np.abs(lat).max()) * float(cell_size_m)
    # Orient +e1 to a non-negative row component (tie: non-negative col) so the
    # vertex order is reproducible; tools orient relative to the anchor.
    if (e1[0] < 0) or (e1[0] == 0 and e1[1] < 0):
        s = -s
    bins = np.round(s).astype(int)
    ub, inv = np.unique(bins, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    if len(ub) < 2:
        return None
    sums = np.zeros((len(ub), 2), dtype=float)
    cnt = np.zeros(len(ub), dtype=float)
    np.add.at(sums, inv, pts)
    np.add.at(cnt, inv, 1.0)
    cl = sums / cnt[:, None]  # (n_bins, 2), ascending by projection
    seg = np.hypot(np.diff(cl[:, 0]), np.diff(cl[:, 1])) * float(cell_size_m)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    length_m = float(arc[-1])
    if length_m <= 0.0:
        return None
    return (
        [[round(float(r), 2), round(float(c), 2)] for r, c in cl],
        [round(float(v), 2) for v in arc],
        round(length_m, 2),
        round(residual_m, 2),
        "pca_projection",
    )


def _build_node(
    fid: str, class_name: str, feat: dict, *, cell_size_m: float
) -> MapObject | None:
    """Build a MapObject from a baked feature dict. Returns None if no cells."""
    cells = feat["cells"]
    if not cells:
        return None

    arr = np.array(cells)
    centroid = (int(round(feat["centroid_row"])), int(round(feat["centroid_col"])))

    angle, confidence = _compute_orientation(arr[:, 0], arr[:, 1])
    # The mask centerline family, the same for every class.
    _geom = _mask_centerline(arr, cell_size_m=cell_size_m)
    if _geom is not None:
        _centerline, _arc_m, _axis_len_m, _axis_resid_m, _cl_method = _geom
    else:
        _centerline = None
        _arc_m = None
        _axis_len_m = 0.0
        _axis_resid_m = 0.0
        _cl_method = ""

    # The bake's exact world-meter vertex chain (a ring, a way, or a node's
    # single vertex), stamped verbatim; the bake omits it when the geometry
    # has none.
    world_contour = [[float(p[0]), float(p[1])] for p in feat.get("world_contour") or []] or None

    return MapObject(
        id=fid,
        class_name=class_name,
        name=feat.get("name"),
        tags=_extract_tags(feat),
        centroid=centroid,
        bbox=_compute_bbox(cells),
        cell_count=len(cells),
        orientation=angle,
        orientation_confidence=confidence,
        cells=arr.astype(np.int32, copy=False),
        world_contour=world_contour,
        centerline=_centerline,
        centerline_arc_m=_arc_m,
        axis_length_m=_axis_len_m,
        axis_residual_m=_axis_resid_m,
        centerline_method=_cl_method,
    )


def build_object_graph(
    features: dict,
    *,
    cell_size_m: float,
) -> ObjectGraph:
    """Build the object graph (nodes only) from features.json data.

    Emits one ``MapObject`` per feature — centroid, bbox, cells, orientation,
    centerline. No edges: spatial relations are derived geometrically from
    node xy/cells at query time by the geom_compute ops.
    """
    nodes: dict[str, MapObject] = {}

    for class_name, feats in features.items():
        if class_name in _NON_FEATURE_KEYS:
            continue
        if not isinstance(feats, list):
            continue
        for i, feat in enumerate(feats):
            if not isinstance(feat, dict):
                continue
            # Fids carry provenance, never semantics: ``<features-key>_<i>``
            # (the bake writes one ``prior`` list, so ``prior_<i>``, i stable
            # per bake sort order), mirroring the runtime's ``detected_<i>``.
            fid = f"{class_name}_{i}"
            node = _build_node(fid, class_name, feat, cell_size_m=cell_size_m)
            if node is not None:
                nodes[fid] = node

    logging.info("Graph complete: %d nodes", len(nodes))
    return ObjectGraph(nodes=nodes, cell_size_m=float(cell_size_m))
