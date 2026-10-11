"""Catalog-prior footprints projected onto the ortho crop.

:func:`build_prior_polys_ortho` rasterizes each requested prior's tile-grid
footprint onto the crop's pixels; the per-stop prior colour enrichment
(``benchmarks.citynav.run_benchmark.osm_prior``) reads M6 colour inside each
mask, the same reading SAM detections get.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from maps.perception.geometry.geolocate import TileFrame
from maps.perception.ortho.sensor_frame import frame_corners_px
from maps.util.mask_region import MaskRegion


def _fill_region(
    polys: "list[np.ndarray]", *, img_h: int, img_w: int
) -> "MaskRegion | None":
    """Fill pixel-space polygons into a region covering only what they touch.

    ``cv2.fillPoly`` clips to whatever array it is handed, so translating the
    polygons into a box clipped to the frame draws exactly the pixels a
    frame-sized fill would have drawn, in the same places. Integer translation
    is exact, so the rasterisation is unchanged and not merely close.

    ``None`` when the polygons touch no pixel inside the frame.
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


def _rasterize_cells_to_mask(
    cells: Sequence[Sequence[int]],
    bounds: TileFrame,
    world_to_px: np.ndarray,
    img_h: int,
    img_w: int,
) -> "MaskRegion | None":
    """Project a feature's tile-grid cells onto the image plane, all at once.

    Each cell is a square of side ``bounds.cell_size_m``; its corners (BL, BR,
    TR, TL) go through ``world_to_px`` in one ``(N, 4, 3)`` matmul, a cell
    whose corner lands on the line at infinity is dropped, and the quads are
    filled into one region (:func:`_fill_region`). A Cal City stop has ~100
    priors in view, so this runs per prior on every stop.
    """
    a = np.asarray(cells, dtype=np.float64).reshape(-1, 2)
    if a.size == 0:
        return None
    cs = float(bounds.cell_size_m)
    half = cs / 2.0
    cx = bounds.x_min + (a[:, 1] + 0.5) * cs
    cy = bounds.y_min + (a[:, 0] + 0.5) * cs
    dx = np.array([-half, half, half, -half], dtype=np.float64)
    dy = np.array([-half, -half, half, half], dtype=np.float64)
    xs = cx[:, None] + dx[None, :]
    ys = cy[:, None] + dy[None, :]
    proj = np.stack([xs, ys, np.ones_like(xs)], axis=-1) @ world_to_px.astype(np.float64, copy=False).T
    denom = proj[..., 2]
    keep = np.all(np.isfinite(denom), axis=1) & ~np.any(np.abs(denom) < 1e-9, axis=1)
    quads = np.stack([proj[..., 0] / denom, proj[..., 1] / denom], axis=-1).astype(np.int32)
    return _fill_region([quads[i] for i in np.nonzero(keep)[0]], img_h=img_h, img_w=img_w)


def build_prior_polys_ortho(
    *,
    rgb_shape: tuple[int, int],
    features: Mapping[str, Sequence[Mapping[str, Any]]],
    bounds: TileFrame,
    view_corners_xy: Sequence[tuple[float, float]],
    fids_to_render: Sequence[str],
    target_fid: str | None = None,
) -> tuple[list[dict[str, Any]], list]:
    """The requested priors' footprints as masks on the ortho crop.

    ``features`` is the bake's per-class ``base_features``; a row's id is its
    ``fid``, else ``cityref_<cityref_id>``, else ``<class>_<index>`` (the
    runtime graph's positional id). Each row in ``fids_to_render`` with a
    centroid and cells is rasterized through the ``world -> pixel``
    homography of ``view_corners_xy`` (``[FL, FR, BR, BL]``) at ``rgb_shape``;
    one that touches no pixel is skipped.

    Returns ``(polys, [])``: ``polys`` is ``[{"fid", "mask"}]``, ``target_fid``
    first, then by short id (``C<cityref_id>``, else the fid).
    """
    img_h, img_w = int(rgb_shape[0]), int(rgb_shape[1])
    src = np.array([(float(c[0]), float(c[1])) for c in view_corners_xy], dtype=np.float32)
    world_to_px = cv2.getPerspectiveTransform(src, frame_corners_px(img_w, img_h))
    wanted = {str(f).strip() for f in (fids_to_render or []) if f}
    keyed = []
    for class_name, feats in features.items():
        for idx, feat in enumerate(feats):
            cid = feat.get("cityref_id")
            if feat.get("fid") is not None:
                fid = str(feat["fid"])
            elif cid is not None:
                fid = f"cityref_{cid}"
            else:
                fid = f"{class_name}_{idx}"
            if fid not in wanted or feat.get("centroid_row") is None or feat.get("centroid_col") is None:
                continue
            mask = _rasterize_cells_to_mask(feat.get("cells") or [], bounds, world_to_px, img_h, img_w)
            if mask is None:
                continue
            short_id = f"C{cid}" if cid is not None else fid
            keyed.append(((fid != target_fid, short_id), {"fid": fid, "mask": mask}))
    keyed.sort(key=lambda kp: kp[0])
    # ponytail: the empty second slot is osm_prior's two-value unpack; it goes when osm_prior takes one value
    return [poly for _, poly in keyed], []
