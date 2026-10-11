"""Per-frame prior colour enrichment: the bake's prior footprints in view,
projected through the observation's view square and coloured by the same M6
extractor detections get. Colour is ORDERING-only: it feeds score_by, never
a filter.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from maps.perception.viewpoint.base import TileBounds
from maps.perception.viewpoint.overlay import build_prior_polys_ortho
from maps.perception.viz.color import color_from_mask_m6

log = logging.getLogger("run_benchmark")


def _prior_fids_in_view(assets, view_corners_xy) -> list[str]:
    """The prior fids whose centroid lands in the view square's bbox, widened
    by 10% + 5 m so a footprint straddling the edge still enriches; ``[]``
    with no view square. A fid is the graph's positional ``<class>_<index>``
    (the bake carries no explicit fid)."""
    if not view_corners_xy:
        return []
    xs = [float(c[0]) for c in view_corners_xy]
    ys = [float(c[1]) for c in view_corners_xy]
    mx = (max(xs) - min(xs)) * 0.1 + 5.0
    my = (max(ys) - min(ys)) * 0.1 + 5.0
    x0, x1, y0, y1 = min(xs) - mx, max(xs) + mx, min(ys) - my, max(ys) + my
    cs, bx, by = float(assets.cell_size_m), float(assets.bounds.x_min), float(assets.bounds.y_min)
    out: list[str] = []
    for class_name, feats in (assets.base_features or {}).items():
        for idx, feat in enumerate(feats):
            cr, cc = feat.get("centroid_row"), feat.get("centroid_col")
            if cr is None or cc is None:
                continue
            if x0 <= bx + (float(cc) + 0.5) * cs <= x1 and y0 <= by + (float(cr) + 0.5) * cs <= y1:
                out.append(f"{class_name}_{idx}")
    return out


def _build_frame_enrich(*, obs, step_idx: int, view_corners, assets, build_prior: bool) -> dict:
    """One frame's prior enrichment, everything but the graph apply:
    ``{"prior": [(fid, colour attrs, mask px)], "prior_exc", "phase_s"}``.
    A failure is carried in ``prior_exc`` with the items done before it, so
    the flight's apply warns and counts it."""
    out: dict = {"prior": [], "prior_exc": None}
    # The steps of the one enrich mark in the stop's phase line.
    _t: dict = {}
    _c = time.perf_counter
    rgb_arr = np.asarray(obs.rgb)
    if build_prior:
        try:
            _m = _c()
            enrich_fids = _prior_fids_in_view(assets, view_corners)
            _t["p_fids"] = _c() - _m
            _t["p_n"] = float(len(enrich_fids))
            _m = _c()
            if rgb_arr.ndim == 3 and enrich_fids:
                try:
                    prior_polys, _table = build_prior_polys_ortho(
                        rgb_shape=(int(rgb_arr.shape[0]), int(rgb_arr.shape[1])),
                        features=assets.base_features,
                        bounds=TileBounds(x_min=float(assets.bounds.x_min), y_min=float(assets.bounds.y_min),
                                          cell_size_m=float(assets.cell_size_m),
                                          grid_shape=tuple(assets.grid_shape)),
                        view_corners_xy=view_corners,
                        fids_to_render=enrich_fids,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("catalog prior overlay build failed step=%d anchor=?: %s", step_idx, exc)
                    prior_polys = None
                _t["p_polys"] = _c() - _m
                _m = _c()
                for poly in prior_polys or []:
                    pfid = str(poly.get("fid") or "")
                    pmask = poly.get("mask")
                    if not pfid or pmask is None:
                        continue
                    # One colour contract: priors get M6 over the projected
                    # footprint, the extractor detections get.
                    mix = color_from_mask_m6(rgb_arr, pmask)
                    if not mix:
                        continue
                    pattrs = {
                        "color": mix["color"],
                        "color_conf": f"{mix['color_conf']:.3f}",
                        "color_shade": mix["color_shade"],
                        "color_flag": mix["color_flag"],
                    }
                    out["prior"].append((pfid, pattrs, pmask.area))
                _t["p_color"] = _c() - _m
        except Exception as exc:  # noqa: BLE001 - carried to the apply
            out["prior_exc"] = exc
    out["phase_s"] = _t
    return out
