"""Catalog-prior footprint -> image-mask projection builders.

Projects a prior feature's tile-grid (or world-contour) footprint onto a
frame's image plane and returns per-feature masks + matching table rows.
These masks are the projection reused by:

- the per-stop prior COLOR ENRICHMENT (``benchmarks.citynav.run_benchmark`` reads a
  perceived color off the rendered pixels inside each in-view prior's mask,
  the SAME postproc SAM detections get, and merges it onto the prior node);

Builders:

- :func:`build_prior_polys_ortho` (canonical): per-feature masks + table
  rows for the ortho (``ortho``) crop, sourced from the NAMED
  ``base_features`` tile-grid footprints.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from maps.util.mask_region import MaskRegion

from maps.perception.viewpoint.base import TileBounds


def _fill_region(
    polys: "list[np.ndarray]", *, img_h: int, img_w: int
) -> "MaskRegion | None":
    """Fill pixel-space polygons into a region covering only what they touch.

    Both prior builders used to allocate one FULL-resolution array per
    feature — ~76 features a stop, each a 2016x2016 array for a building
    covering a fraction of it — which was the largest client-side cost left
    once the detection path went crop-native.

    ``cv2.fillPoly`` clips to whatever array it is handed, so translating the
    polygons into a box clipped to the frame draws exactly the pixels a
    frame-sized fill would have drawn, in the same places. Integer translation
    is exact, so the rasterisation is unchanged and not merely close.

    ``None`` when the polygons touch no pixel inside the frame — the case the
    callers used to spot with ``mask.any()``.
    """
    if not polys:
        return None
    allpts = np.concatenate(polys, axis=0)
    c0 = max(0, int(allpts[:, 0].min()))
    r0 = max(0, int(allpts[:, 1].min()))
    c1 = min(img_w - 1, int(allpts[:, 0].max()))
    r1 = min(img_h - 1, int(allpts[:, 1].max()))
    if r1 < r0 or c1 < c0:
        return None  # projects entirely outside the view square
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.uint8)
    offset = np.array([c0, r0], dtype=np.int32)
    cv2.fillPoly(sub, [p - offset for p in polys], 1)
    crop = sub.astype(bool)
    if not crop.any():
        return None
    return MaskRegion(crop, (r0, c0, r1, c1), (img_h, img_w))


def _project_cell_quads(cells, bounds, h_t, cs, half) -> "list[np.ndarray] | None":
    """Every cell's pixel quad, in one projection. ``None`` if ``cells`` is
    not a numeric ``(N, 2)`` block, which is the caller's cue to walk it."""
    try:
        a = np.asarray(cells, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if a.size == 0:
        return []
    cx = bounds.x_min + (a[:, 1] + 0.5) * cs
    cy = bounds.y_min + (a[:, 0] + 0.5) * cs
    # Corner order is the per-cell form's, verbatim: BL, BR, TR, TL.
    dx = np.array([-half, half, half, -half], dtype=np.float64)
    dy = np.array([-half, -half, half, half], dtype=np.float64)
    xs = cx[:, None] + dx[None, :]
    ys = cy[:, None] + dy[None, :]
    pts_h = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
    proj = pts_h @ h_t.T
    denom = proj[..., 2]
    keep = np.all(np.isfinite(denom), axis=1) & ~np.any(
        np.abs(denom) < 1e-9, axis=1
    )
    quads = np.stack([proj[..., 0] / denom, proj[..., 1] / denom],
                     axis=-1).astype(np.int32)
    return [quads[i] for i in np.nonzero(keep)[0]]


def _project_cell_quads_one_by_one(cells, bounds, h_t, cs, half) -> list:
    """The same projection, one malformed-tolerant cell at a time."""
    quads: list[np.ndarray] = []
    for cell in cells:
        try:
            r = float(cell[0])
            c = float(cell[1])
        except (TypeError, IndexError, ValueError):
            continue
        cx = bounds.x_min + (c + 0.5) * cs
        cy = bounds.y_min + (r + 0.5) * cs
        corners_w = np.array(
            [
                [cx - half, cy - half],
                [cx + half, cy - half],
                [cx + half, cy + half],
                [cx - half, cy + half],
            ],
            dtype=np.float64,
        )
        pts_h = np.column_stack([corners_w, np.ones(4)])
        proj = pts_h @ h_t.T
        denom = proj[:, 2]
        if not np.all(np.isfinite(denom)) or np.any(np.abs(denom) < 1e-9):
            continue
        quads.append(
            np.column_stack([proj[:, 0] / denom, proj[:, 1] / denom]).astype(
                np.int32
            )
        )
    return quads


def _rasterize_cells_to_mask(
    cells: Sequence[Sequence[int]],
    bounds: TileBounds,
    world_to_px: np.ndarray,
    img_h: int,
    img_w: int,
) -> "MaskRegion | None":
    """Project a feature's tile-grid cell list onto the image plane.

    Each cell is a unit square of side ``bounds.cell_size_m`` in world
    meters; its four corners project through ``world_to_px`` (the
    homography ``cv2.getPerspectiveTransform(view_corners_xy ->
    image_corners)``) to a quadrilateral in pixel space. ``cv2.fillPoly``
    unions the per-cell quads into a single binary mask suitable for
    ``cv2.findContours``.

    ALL CELLS AT ONCE. This runs once per prior in view and there are ~100 of
    those per stop, so a Python loop over cells was doing ~63,000 four-row
    allocations and four-row matmuls per stop: measured at 1.13 s of a 4.3 s
    Cal City stop, the largest single cost left in one. Projecting the whole
    ``(N, 4, 3)`` corner block in one matmul is 19x faster and produces
    byte-identical quads -- checked against the per-cell form on 472,470 real
    cells across 500 Cal City footprints, 0 differing.
    """
    if cells is None or len(cells) == 0:
        return np.zeros((img_h, img_w), dtype=bool)
    cs = float(bounds.cell_size_m)
    half = cs / 2.0
    h_t = world_to_px.astype(np.float64, copy=False)
    quads = _project_cell_quads(cells, bounds, h_t, cs, half)
    if quads is None:
        # Ragged or non-numeric cells: fall back to the per-cell walk, which
        # drops the malformed entries one at a time and keeps the rest. The
        # bake writes uniform (N, 2) integer arrays, so this is a corrupt-input
        # path, not a second implementation of the normal one.
        quads = _project_cell_quads_one_by_one(cells, bounds, h_t, cs, half)
    if not quads:
        return None
    # Fill only the box the quads cover. A prior building is a small part of
    # the frame, and one full-resolution array per feature (~76 features a
    # stop) was the largest client-side cost left after the detection path
    # went crop-native. fillPoly clips to whatever array it is given, so
    # translating the quads into a box clipped to the frame draws exactly the
    # pixels a frame-sized fill would have.
    return _fill_region(quads, img_h=img_h, img_w=img_w)


def build_prior_polys_ortho(
    *,
    rgb_shape: tuple[int, int],
    features: Mapping[str, Sequence[Mapping[str, Any]]],
    bounds: TileBounds,
    view_corners_xy: Sequence[tuple[float, float]],
    fids_to_render: Sequence[str],
    target_fid: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the catalog-prior polygon set for the ortho per-stop crop.

    Selects the catalog features whose canonical ``fid`` appears in
    ``fids_to_render``, rasterizes each one's tile-grid footprint to an
    image-resolution mask via the ``world -> pixel`` homography
    (``cv2.getPerspectiveTransform(view_corners_xy -> image_corners)``), and
    packages what the per-stop prior color enrichment needs to read pixels
    inside each mask.

    Args:
        rgb_shape: ``(H, W)`` of the canvas the overlay will be rendered
            on. The masks come out at this resolution.
        features: per-class lists of feature dicts (the ``base_features``
            mapping that ``benchmarks.citynav.run_benchmark`` loads from
            ``<block>_features.json``).
        bounds: tile-grid layout (``x_min`` / ``y_min`` / ``cell_size_m``
            / ``grid_shape``) for the active block or city aggregate.
        view_corners_xy: the upstream ``[FL, FR, BR, BL]`` world corners
            of the crop (``Observation.ortho_view_corners_xy``).
        fids_to_render: feature ids to draw. The runtime
            union (target + every leaf's ``objects_to_mask``) lands
            here; ids that resolve to no feature row are silently
            skipped (a single ``unresolved_prior_fid`` log line is the
            caller's responsibility).
        target_fid: optional feature id whose row gets
            ``is_target=True`` and a ``TARGET `` label prefix.

    Returns:
        ``(prior_polys, cityref_table)``:

        - ``prior_polys`` is a list of ``{cityref_id, fid, short_id,
          class_name, name, world_xy, pixel_centroid, mask, label,
          is_target}`` dicts; ``mask`` is a ``(H, W)`` bool array
          (resolution-matched to ``rgb_shape``).
        - ``cityref_table`` carries the same row shape minus the ``mask``
          field (``n_hit_cells`` / ``n_hit_pixels`` / ``ray_length_m_mean``
          zeroed for parity with the perspective path).
    """
    img_h, img_w = int(rgb_shape[0]), int(rgb_shape[1])

    src = np.array(
        [(float(c[0]), float(c[1])) for c in view_corners_xy], dtype=np.float32
    )
    dst = np.array(
        [(0, 0), (img_w - 1, 0), (img_w - 1, img_h - 1), (0, img_h - 1)],
        dtype=np.float32,
    )
    world_to_px = cv2.getPerspectiveTransform(src, dst)

    cs = float(bounds.cell_size_m)
    wanted: set[str] = {str(f).strip() for f in (fids_to_render or []) if f}

    prior_polys: list[dict[str, Any]] = []
    table: list[dict[str, Any]] = []

    for class_name, feats in features.items():
        for idx, feat in enumerate(feats):
            explicit_fid = feat.get("fid")
            cid = feat.get("cityref_id")
            if explicit_fid is not None:
                fid = str(explicit_fid)
            elif cid is not None:
                fid = f"cityref_{cid}"
            else:
                # The citynav bake keys building/road/etc rows only by name
                # (no ``fid`` / ``cityref_id``); the runtime ObjectGraph
                # assigns "<class>_<index>" by position. Mirror that here, or
                # the leaf ``objects_to_mask`` fids (building_20, road_3, ...)
                # match NO feature row and NO candidate is ever outlined on
                # the per-stop crop (has_prior_overlay stays False — the
                # reasoner reads an unmarked building and its color verdicts
                # are ungrounded guesses).
                fid = f"{class_name}_{idx}"
            if fid not in wanted:
                continue
            cr = feat.get("centroid_row")
            cc = feat.get("centroid_col")
            if cr is None or cc is None:
                continue
            wx = bounds.x_min + (float(cc) + 0.5) * cs
            wy = bounds.y_min + (float(cr) + 0.5) * cs
            # A centroid on the homography's line at infinity has no pixel.
            denom = world_to_px[2, 0] * wx + world_to_px[2, 1] * wy + world_to_px[2, 2]
            if denom == 0:
                continue
            mask = _rasterize_cells_to_mask(
                feat.get("cells") or [], bounds, world_to_px, img_h, img_w
            )
            if mask is None:
                # The polygon projects entirely outside the view square;
                # skip both the overlay and the text row.
                continue
            # Prefer the mask's pixel centroid for label placement so the
            # text lands inside the visible portion when the polygon is
            # clipped at the frame edge.
            ys, xs = mask.rows_cols()
            cx_label = int(xs.mean())
            cy_label = int(ys.mean())
            is_target = fid == target_fid
            short_id = f"C{cid}" if cid is not None else fid
            name = feat.get("name") or ""
            # Names shown consistently on target + anchors: the catalog name
            # is admissible prior (the map's name, not the operator GT), and
            # anchors NEED their name so the reasoner can map an outlined
            # feature to the named entity in the question (spatial reasoning). cv2
            # FONT_HERSHEY can't render the Unicode middle dot; plain space.
            name_suffix = f" {name}" if name else ""
            label_text = f"{short_id}{name_suffix}"
            if is_target:
                label_text = f"TARGET {label_text}"
            row: dict[str, Any] = {
                "cityref_id": cid,
                "fid": fid,
                "short_id": short_id,
                "class_name": class_name,
                "name": feat.get("name"),
                "world_xy": (wx, wy),
                "pixel_centroid": (cx_label, cy_label),
                "n_hit_cells": 0,
                "n_hit_pixels": 0,
                "ray_length_m_mean": None,
                "is_target_anchor": is_target,
            }
            poly: dict[str, Any] = dict(row)
            poly["mask"] = mask
            poly["label"] = label_text
            poly["is_target"] = is_target
            prior_polys.append(poly)
            table.append(row)

    # Deterministic ordering: target first, then short_id ascending. The
    # renderer draws polygons in list order, so the target's fill +
    # outline land first and any later overlap respects its presence.
    prior_polys.sort(key=lambda p: (not p["is_target"], p["short_id"]))
    table.sort(key=lambda r: (not r["is_target_anchor"], r["short_id"]))
    return prior_polys, table
