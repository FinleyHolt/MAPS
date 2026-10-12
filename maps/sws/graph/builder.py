"""The prebake graph: one ``MapObject`` per baked feature (centroid, cells,
centerline, contour, tags)."""

from __future__ import annotations

import logging

import numpy as np

from maps.sws.graph.model import MapObject, ObjectGraph

# Keys in the ``features`` dict that carry non-feature metadata rather than a
# feature list. ``build_object_graph`` skips them.
_NON_FEATURE_KEYS: frozenset[str] = frozenset(
    {"bbox", "georef", "_tile_name", "nlcd_zones"}
)


# Feature keys that never become tags: the geometry the builder reads, ``name``
# (its own field), OSM's ``orientation`` (not served) and ``descriptions``
# (the GT-label wall). Everything else rides into ``MapObject.tags`` verbatim.
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


def _mask_centerline(
    arr: np.ndarray, *, cell_size_m: float
) -> tuple[list[list[float]], float] | None:
    """A mask's principal-axis centerline: ``(centerline_rc, length_m)``, or
    ``None`` for a mask too small to have an axis.

    Cells are projected onto the PCA axis and binned per grid step; each bin
    contributes its mean cell (``(row, col)`` vertices).
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
    s = centered @ e1  # signed projection along the principal axis (cells)
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
    length_m = float(np.concatenate([[0.0], np.cumsum(seg)])[-1])
    if length_m <= 0.0:
        return None
    return [[round(float(r), 2), round(float(c), 2)] for r, c in cl], round(length_m, 2)


def _build_node(
    class_name: str, feat: dict, *, cell_size_m: float
) -> MapObject | None:
    """Build a MapObject from a baked feature dict. Returns None if no cells."""
    cells = feat["cells"]
    if not cells:
        return None

    arr = np.array(cells)
    centroid = (int(round(feat["centroid_row"])), int(round(feat["centroid_col"])))

    # The mask centerline, the same for every class.
    centerline, axis_length_m = _mask_centerline(arr, cell_size_m=cell_size_m) or (None, 0.0)

    # The bake's exact world-meter vertex chain (a ring, a way, or a node's
    # single vertex), stamped verbatim; the bake omits it when the geometry
    # has none.
    world_contour = [[float(p[0]), float(p[1])] for p in feat.get("world_contour") or []] or None

    return MapObject(
        class_name=class_name,
        name=feat.get("name"),
        tags=_extract_tags(feat),
        centroid=centroid,
        cells=arr.astype(np.int32, copy=False),
        world_contour=world_contour,
        centerline=centerline,
        axis_length_m=axis_length_m,
    )


def build_object_graph(
    features: dict,
    *,
    cell_size_m: float,
) -> ObjectGraph:
    """The nodes-only object graph of a ``features.json``."""
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
            node = _build_node(class_name, feat, cell_size_m=cell_size_m)
            if node is not None:
                nodes[fid] = node

    logging.info("Graph complete: %d nodes", len(nodes))
    return ObjectGraph(nodes=nodes, cell_size_m=float(cell_size_m))
