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
# list of OSM / CityRefer features. ``build_object_graph`` skips them.
_NON_FEATURE_KEYS: frozenset[str] = frozenset(
    {"bbox", "georef", "_tile_name", "nlcd_zones"}
)


# Geometry / bookkeeping keys stored on dedicated MapObject fields (or
# consumed by the builder itself), never as tags. Everything else on a feature
# rides into ``MapObject.tags`` verbatim — the agent pulls semantics from the
# tags at runtime; there is no semantic tag denylist. ``descriptions`` is the
# GT-label wall, stored on ``MapObject.descriptions`` for non-LLM
# consumers only.
_TAG_SKIP_KEYS = frozenset(
    {
        "cells",
        "cell",
        "world_contour",  # stamped on MapObject.world_contour, not a tag
        "centroid_row",
        "centroid_col",
        "centroid_lat",
        "centroid_lon",
        "lat",
        "lon",
        "orientation",
        "name",  # stored separately on MapObject
        "descriptions",  # stored on MapObject.descriptions, not in tags
        "cityref_id",  # explicitly added to tags below; skip generic scrape
        "object_type",
        "altitude_m",
        "height_m",
    }
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


def _get_cells(feat: dict) -> list[list[int]]:
    """Get cell coordinates from a feature."""
    if "cells" in feat:
        return feat["cells"]
    if "cell" in feat:
        return [feat["cell"]]
    return []


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

    The centerline is the mask's medial line sampled at the grid resolution
    along its PCA principal axis: project every cell onto the axis, bin by the
    integer projection step (data-driven — the grid's own resolution, no
    tunable bin count), and take the lateral mask-midpoint (mean cell
    position) per bin. Everything is in the ``(row, col)`` grid frame (same as
    ``centroid``/``cells``); the meter scalars are frame-invariant. No compass
    bearing or signed side is computed here — those are frame-sensitive and are
    derived at the tool boundary (see the frame-convention geometry in
    maps.search ``geom_compute``), never baked.

    ``residual_m`` is the max perpendicular distance (meters) of any mask cell
    to the straight PCA line — a curvature/fit beacon (≈ half-width for a
    straight mask, inflated by curvature; the visualizer is authoritative).
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
    # Deterministic axis orientation (a labeling convention, NOT a threshold):
    # orient +e1 to have a non-negative row component (tie-break: non-negative
    # col), so the vertex order is reproducible. Exposed via vertex order only;
    # tools never assume "forward" from it — they orient relative to the anchor.
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
    """Build a MapObject from a feature dict. Returns None if no cells."""
    cells = _get_cells(feat)
    if not cells:
        return None

    arr = np.array(cells)

    # Use precomputed centroid if available
    if "centroid_row" in feat and "centroid_col" in feat:
        centroid = (int(round(feat["centroid_row"])), int(round(feat["centroid_col"])))
    else:
        centroid = (int(round(arr[:, 0].mean())), int(round(arr[:, 1].mean())))

    angle, confidence = _compute_orientation(arr[:, 0], arr[:, 1])
    # Raw per-node geometry: the mask centerline + arc + curvature beacon,
    # computed generically for every node (no class special-casing).
    _geom = _mask_centerline(arr, cell_size_m=cell_size_m)
    if _geom is not None:
        _centerline, _arc_m, _axis_len_m, _axis_resid_m, _cl_method = _geom
    else:
        _centerline = None
        _arc_m = None
        _axis_len_m = 0.0
        _axis_resid_m = 0.0
        _cl_method = ""

    # Read pre-computed WGS84 centroid if available
    geo = None
    if "centroid_lat" in feat and "centroid_lon" in feat:
        geo = (feat["centroid_lat"], feat["centroid_lon"])
    elif "lat" in feat and "lon" in feat:  # POIs use lat/lon directly
        geo = (feat["lat"], feat["lon"])

    descriptions_raw = feat.get("descriptions") or []
    descriptions = [str(d) for d in descriptions_raw if d]

    # The feature's EXACT world-meter vertex chain from the bake — a ring, a
    # way, or a node's single vertex (benchmarks/citynav/bake/city.py).
    # Stamped verbatim onto MapObject.world_contour; build_surface prefers it
    # over the 2 m cell-quantized boundary and computes the shape scalars
    # from it (same substrate a SAM detection's pixel-mask contour uses).
    # NO length floor: a >= 3 floor is a footprint assumption, and it dropped
    # every way and every node the bake now hands over.
    wc_raw = feat.get("world_contour")
    world_contour: list[list[float]] | None = None
    if isinstance(wc_raw, list) and wc_raw:
        try:
            world_contour = [[float(p[0]), float(p[1])] for p in wc_raw]
        except (TypeError, ValueError, IndexError):
            world_contour = None

    tags = _extract_tags(feat)
    # CityRefer features carry these as scalar fields; surface them as tags
    # so downstream code can read them without a special-case schema check.
    if "cityref_id" in feat and feat["cityref_id"] is not None:
        tags["cityref_id"] = str(feat["cityref_id"])
    if "object_type" in feat and feat["object_type"]:
        tags.setdefault("object_type", str(feat["object_type"]))
    if "altitude_m" in feat and feat["altitude_m"] is not None:
        tags["altitude_m"] = f"{float(feat['altitude_m']):.3f}"
    if "height_m" in feat and feat["height_m"] is not None:
        tags["height_m"] = f"{float(feat['height_m']):.3f}"

    return MapObject(
        id=fid,
        class_name=class_name,
        name=feat.get("name"),
        tags=tags,
        centroid=centroid,
        bbox=_compute_bbox(cells),
        cell_count=len(cells),
        orientation=angle,
        orientation_confidence=confidence,
        geo=geo,
        area_m2=float(len(cells)) * float(cell_size_m) * float(cell_size_m),
        descriptions=descriptions,
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
            # Fids carry PROVENANCE, never semantics. CityRefer features carry
            # a stable dataset object id — prefer `cityref_<id>` so the
            # dataset's reference (used by episodes' ``object_ids``) is the
            # canonical handle through the graph. Every other baked feature is
            # ``<features-key>_<i>`` — the bake writes one ``prior`` list, so
            # priors mint ``prior_<i>``, mirroring the runtime's
            # ``detected_<i>`` (provenance_index; i stable per bake sort
            # order). All semantics live on the row (name + tags).
            cityref_id = feat.get("cityref_id")
            if cityref_id is not None:
                # str form supports block-namespaced ids in the city-level
                # aggregate (``birmingham_block_0_3``); int is the legacy
                # per-block shape. Both pass through unchanged.
                if isinstance(cityref_id, str):
                    fid = f"cityref_{cityref_id}"
                else:
                    fid = f"cityref_{int(cityref_id)}"
            else:
                fid = f"{class_name}_{i}"
            node = _build_node(fid, class_name, feat, cell_size_m=cell_size_m)
            if node is not None:
                nodes[fid] = node

    # Per-class object sets O_n^{(c)}.
    objects_by_class: dict[str, dict[str, MapObject]] = {}
    for fid, node in nodes.items():
        objects_by_class.setdefault(node.class_name, {})[fid] = node

    logging.info("Graph complete: %d nodes", len(nodes))
    return ObjectGraph(
        nodes=nodes,
        cell_size_m=float(cell_size_m),
        objects_by_class=objects_by_class,
    )
