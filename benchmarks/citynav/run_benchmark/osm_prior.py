"""Ortho OSM prior projection + color enrichment. Color is
ORDERING-only (CLAUDE.md): the enrichment feeds score_by, never a filter.
"""

from __future__ import annotations

from maps.perception.viz.color import color_from_mask_m6
import time

import numpy as np
import logging

log = logging.getLogger("run_benchmark")


def _maybe_build_prior_polys(
    *,
    rgb_shape: tuple[int, int],
    assets,
    view_corners_xy,
    fids_to_render,
    target_fid: str,
    step_idx: int,
) -> tuple[list[dict] | None, list[dict] | None]:
    """Build the catalog-prior polygon set for the per-stop vision image.

    Returns ``(prior_polys, cityref_table)``. ``prior_polys`` carries
    per-feature masks for
    the prior overlay; ``cityref_table``
    mirrors the row shape the text-table renderer expects. Both
    are filtered to the same ``fids_to_render`` set (the union of every
    leaf's ``objects_to_mask`` plus the anchor) so the visual and text
    surfaces the reasoner reads stay in lockstep. The footprints are the
    tile grid's, rasterized through the observation's view square by
    :func:`build_prior_polys_ortho`. Returns ``(None, None)`` when
    ``fids_to_render`` is empty.
    """
    if not fids_to_render:
        return None, None
    try:
        if assets is None:
            return None, None
        from maps.perception.viewpoint.base import TileBounds as _CRPriorBounds
        from maps.perception.viewpoint.overlay import build_prior_polys_ortho

        cr_bounds = _CRPriorBounds(
            x_min=float(assets.bounds.x_min),
            y_min=float(assets.bounds.y_min),
            cell_size_m=float(assets.cell_size_m),
            grid_shape=tuple(assets.grid_shape),
        )
        return build_prior_polys_ortho(
            rgb_shape=rgb_shape,
            features=assets.base_features,
            bounds=cr_bounds,
            view_corners_xy=view_corners_xy,
            fids_to_render=list(fids_to_render),
            target_fid=target_fid or None,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "catalog prior overlay build failed step=%d anchor=%s: %s",
            step_idx,
            target_fid or "?",
            exc,
        )
        return None, None


def _prior_fids_in_view(assets, view_corners_xy) -> list[str]:
    """Prior feature ids plausibly within the current view.

    A cheap pre-filter so the per-frame footprint rasterization (via
    :func:`_maybe_build_prior_polys`) only runs over plausibly in-frame priors,
    not the whole city catalog: prior centroids whose world xy lands within
    the view-square bbox, over ``assets.base_features``, in the runtime graph
    node namespace (explicit ``fid`` -> ``cityref_<id>`` -> ``<class>_<idx>``,
    mirroring :func:`build_prior_polys_ortho`'s fid derivation). Returns
    ``[]`` with no view square."""
    if not view_corners_xy or assets is None:
        return []
    try:
        xs = [float(c[0]) for c in view_corners_xy]
        ys = [float(c[1]) for c in view_corners_xy]
    except (TypeError, ValueError, IndexError):
        return []
    if not xs or not ys:
        return []
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    # Margin so a footprint straddling the view edge still enriches.
    mx = (x1 - x0) * 0.1 + 5.0
    my = (y1 - y0) * 0.1 + 5.0
    x0 -= mx
    x1 += mx
    y0 -= my
    y1 += my
    cs = float(assets.cell_size_m)
    bx = float(assets.bounds.x_min)
    by = float(assets.bounds.y_min)
    out: list[str] = []
    for class_name, feats in (assets.base_features or {}).items():
        for idx, feat in enumerate(feats):
            cr = feat.get("centroid_row")
            cc = feat.get("centroid_col")
            if cr is None or cc is None:
                continue
            wx = bx + (float(cc) + 0.5) * cs
            wy = by + (float(cr) + 0.5) * cs
            if not (x0 <= wx <= x1 and y0 <= wy <= y1):
                continue
            explicit = feat.get("fid")
            cid = feat.get("cityref_id")
            if explicit is not None:
                out.append(str(explicit))
            elif cid is not None:
                out.append(f"cityref_{cid}")
            else:
                out.append(f"{class_name}_{idx}")
    return out


def _build_frame_enrich(
    *,
    obs,
    step_idx: int,
    view_corners,
    assets,
    build_prior: bool,
):
    """Pure per-frame prior-enrich BUILD: in-view fids, poly rasterization
    and color extraction — everything except the graph APPLY, which stays
    serial in the flight body. Reads
    only the frame (obs.rgb, view_corners) and episode-static context
    (assets.base_features). A
    failure is carried (not raised) so the apply replays the serial
    rate-limited warn + fail-count semantics; items completed before the
    failure are returned, matching the serial partial-apply behavior.
    Returns {"prior": [(pfid, pattrs, ppx)...], "prior_exc", "phase_s"}.
    """
    out: dict = {"prior": [], "prior_exc": None}
    # Where this frame's enrichment went. The whole build is one mark in the
    # stop's phase line; these say which of its steps spent it.
    _t: dict = {}
    _c = time.perf_counter
    rgb_arr = np.asarray(obs.rgb)
    if build_prior:
        try:
            _m = _c()
            enrich_fids = _prior_fids_in_view(assets, view_corners)
            _t["p_fids"] = _c() - _m
            _t["p_n"] = float(len(enrich_fids or []))
            _m = _c()
            if rgb_arr.ndim == 3 and enrich_fids:
                prior_polys, _ptable = _maybe_build_prior_polys(
                    rgb_shape=(int(rgb_arr.shape[0]), int(rgb_arr.shape[1])),
                    assets=assets,
                    view_corners_xy=view_corners,
                    fids_to_render=enrich_fids,
                    target_fid="",
                    step_idx=step_idx,
                )
                _t["p_polys"] = _c() - _m
                _m = _c()
                if prior_polys:
                    for poly in prior_polys:
                        pfid = str(poly.get("fid") or "")
                        pmask = poly.get("mask")
                        if not pfid or pmask is None:
                            continue
                        # UNIFIED COLOR CONTRACT: priors get M6 — the SAME
                        # extractor detections get — over the projected OSM
                        # footprint, keyed on best-view pixels, so every
                        # mask, prior or detected, obeys one color contract.
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
