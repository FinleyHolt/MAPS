"""Ortho OSM prior projection + color enrichment. Color is
ORDERING-only (CLAUDE.md): the enrichment feeds score_by, never a filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from maps.perception.viz.color import color_from_mask_m6
from maps.search.analyze_scene.geom_compute import is_closed_chain
import time

import numpy as np
import logging

log = logging.getLogger("run_benchmark")


@dataclass
class _OsmPriorCtx:
    """OSM building/feature contours + meta for the perspective prior overlay.

    Built once per flight_video episode from the block's ``objects.json``;
    keyed ``cityref_<id>`` to match the runtime fid namespace. ``contours_by_fid``
    holds ``(K, 3)`` world (east, north, roof-z) polygons; ``meta_by_fid`` holds
    ``{cityref_id, name, class_name, world_xy}``.
    """

    contours_by_fid: dict[str, np.ndarray]
    meta_by_fid: dict[str, dict]
    # OSM prior fid (``cityref_<id>``) -> runtime-graph node id. Populated once
    # per episode by :func:`_resolve_osm_prior_graph_fids`; empty until then.
    # The headline ViSA bake keys named priors ``<class>_<idx>`` (it strips
    # ``cityref_id``), so the OSM contour namespace never resolves against the
    # runtime graph directly — the perspective enrichment upsert remaps through
    # this map so the perceived color lands on a real node.
    graph_fid_by_fid: dict[str, str] = field(default_factory=dict)


def _build_osm_prior_ctx_from_graph(graph) -> "_OsmPriorCtx | None":
    """Build the prior-color enrichment context from the baked OSM prior nodes.

    The real-stack, CityRefer-FREE source for the prior-enrichment overlay
    (SPEC §5c LEAK L1 fix). The bake stamps ``world_contour`` — the feature's
    exact vertex chain in world meters — on every PRIOR graph node
    (``provenance != "detected"``; also carried in the tile ``features.json``).
    This overlay FILLS that chain to sample colour inside it, so it takes only
    the chains that ENCLOSE (``geom_compute.is_closed_chain``): a way's
    vertices filled as a polygon are a triangle of roadway the feature does not
    have, and the bake now hands over ways and single nodes alongside rings.
    We key the context by each node's OWN graph fid, so
    :attr:`_OsmPriorCtx.graph_fid_by_fid` is the IDENTITY map and NO GT-position
    registration step (:func:`_resolve_osm_prior_graph_fids`) is needed or
    allowed — the enrichment upsert lands directly on the node whose contour we
    projected. A prior with no ``world_contour`` is skipped (no fabrication).

    Contours come out ``(K, 3)`` with z padded 0.0 so the ORTHO projection
    helper (:func:`build_prior_polys_ortho_from_contours`, which reads only
    x/y — z is irrelevant top-down) consumes them unchanged; this is the
    ortho pivot backend. ``world_xy`` is the contour centroid (already
    world meters), which the in-view culls (:func:`_all_osm_fids_in_view`) use.

    Returns ``None`` when ``graph`` carries no contour-bearing priors — the
    caller then disables enrichment under the pivot (honest no-GT) rather than
    reading objects.json.
    """
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict) or not nodes:
        return None
    contours: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    identity: dict[str, str] = {}
    for fid, node in nodes.items():
        if getattr(node, "provenance", None) == "detected":
            continue
        wc = getattr(node, "world_contour", None)
        if not wc:
            continue
        try:
            pts = [(float(p[0]), float(p[1])) for p in wc]
        except (TypeError, ValueError, IndexError):
            continue
        # A fill needs something enclosed. `len >= 3` used to mean "a polygon"
        # because only polygons carried a contour; it now admits a 3-vertex
        # WAY and fills the triangle its bend happens to subtend.
        if not is_closed_chain(pts):
            continue
        fid = str(fid)
        contours[fid] = np.array([[x, y, 0.0] for (x, y) in pts], dtype=float)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        meta[fid] = {
            "cityref_id": None,
            "name": getattr(node, "name", None),
            "class_name": str(getattr(node, "class_name", "") or ""),
            "world_xy": (cx, cy),
        }
        identity[fid] = fid
    if not contours:
        return None
    ctx = _OsmPriorCtx(contours_by_fid=contours, meta_by_fid=meta)
    ctx.graph_fid_by_fid = identity
    log.info(
        "prior-enrichment: sourced %d OSM prior world_contours from the baked "
        "graph (identity fid map; CityRefer-free, SPEC §5c LEAK L1 fix)",
        len(contours),
    )
    return ctx


def _maybe_build_prior_polys(
    *,
    rgb_shape: tuple[int, int],
    assets,
    view_corners_xy,
    fids_to_render,
    target_fid: str,
    step_idx: int,
    obs=None,
    osm_ctx: "_OsmPriorCtx | None" = None,
) -> tuple[list[dict] | None, list[dict] | None]:
    """Build the catalog-prior polygon set for the per-stop vision image.

    Returns ``(prior_polys, cityref_table)``. ``prior_polys`` carries
    per-feature masks for
    the prior overlay; ``cityref_table``
    mirrors the row shape the text-table renderer expects. Both
    are filtered to the same ``fids_to_render`` set (the union of every
    leaf's ``objects_to_mask`` plus the anchor) so the visual and text
    surfaces the reasoner reads stay in lockstep.

    For the orthographic backend (``view_corners_xy`` set) this rasterizes
    the tile-grid footprints via :func:`build_prior_polys_ortho`. For the
    calibrated flight_video perspective backend (``view_corners_xy`` is
    ``None`` but ``obs.sensor`` + ``osm_ctx`` are present) it projects the
    OSM world contours via :func:`build_prior_polys_perspective`, applying
    the per-frame OSM↔SAM registration transform so the prior lands on the
    true building. Returns ``(None, None)`` when ``fids_to_render`` is empty
    or neither backend's inputs are available.
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


def _prior_fids_in_view(
    assets, view_corners_xy, *, osm_ctx: "_OsmPriorCtx | None" = None, obs=None
) -> list[str]:
    """Prior feature ids plausibly within the current view — BACKEND-AGNOSTIC.

    A cheap pre-filter so the per-frame footprint rasterization (via
    :func:`_maybe_build_prior_polys`) only runs over plausibly in-frame priors,
    not the whole city catalog. Returns ids in the runtime graph node
    namespace (``cityref_<id>`` -> explicit ``fid`` -> ``<class>_<idx>``).

    ORTHOGRAPHIC backend (``view_corners_xy`` set, e.g. ortho):
    prior centroids whose world xy lands within the view-square bbox, over
    ``assets.base_features`` (mirrors :func:`build_prior_polys_ortho`'s fid
    derivation).

    PERSPECTIVE backend (``view_corners_xy`` is ``None`` but ``osm_ctx`` +
    ``obs.sensor`` are present): prior centroids
    within a generous ground radius of the camera pose, over
    ``osm_ctx.meta_by_fid``. This is only a coarse cull — the EXACT in-frame
    test happens downstream in :func:`build_prior_polys_perspective`, which
    drops any prior that projects behind / off the camera (empty mask). The
    radius scales with altitude so a plausibly-visible prior is never
    pre-dropped; the projection does the real culling.

    Returns ``[]`` when neither backend's inputs are available (e.g. a
    perspective backend with no ``osm_ctx`` loaded — enrichment then no-ops
    honestly rather than silently mis-firing)."""
    if view_corners_xy:
        if assets is None:
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

    # Perspective backend: coarse pose-radius cull over the OSM prior contours.
    if osm_ctx is None or obs is None or getattr(obs, "sensor", None) is None:
        return []
    pose = getattr(obs, "pose_xy", None)
    if not pose:
        return []
    try:
        px, py = float(pose[0]), float(pose[1])
    except (TypeError, ValueError, IndexError):
        return []
    alt_m = float(getattr(obs, "altitude_m", 0.0) or 0.0)
    fov_m = float(getattr(obs, "fov_m", 0.0) or 0.0)
    # Conservative reach: keep the radius generous and let the projection
    # cull exactly. max(FOV, 6*altitude, 250 m) covers CityNav block scales.
    radius = max(fov_m, alt_m * 6.0, 250.0)
    r2 = radius * radius
    out_p: list[str] = []
    for fid, meta in (osm_ctx.meta_by_fid or {}).items():
        wxy = meta.get("world_xy")
        if not wxy:
            continue
        dx = float(wxy[0]) - px
        dy = float(wxy[1]) - py
        if dx * dx + dy * dy <= r2:
            out_p.append(str(fid))
    return out_p


def _build_frame_enrich(
    *,
    obs,
    step_idx: int,
    view_corners,
    assets,
    osm_ctx,
    build_prior: bool,
):
    """Pure per-frame prior-enrich BUILD: in-view fids, poly rasterization
    and color extraction — everything except the graph APPLY, which stays
    serial in the flight body. Reads
    only the frame (obs.rgb, view_corners) and episode-static context
    (assets.base_features, osm_ctx meta/contours/graph_fid_by_fid). A
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
            enrich_fids = _prior_fids_in_view(
                assets, view_corners, osm_ctx=osm_ctx, obs=obs
            )
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
                    obs=obs,
                    osm_ctx=osm_ctx,
                )
                _t["p_polys"] = _c() - _m
                _m = _c()
                if prior_polys:
                    # Perspective poly fids are OSM ``cityref_<id>``; the
                    # runtime graph keys named priors ``<class>_<idx>`` so
                    # remap through graph_fid_by_fid (named-only footprint
                    # containment) before upsert. Ortho poly fids are
                    # already graph ids; unmapped priors fall through and
                    # the upsert no-ops (never creates a node).
                    gmap = (
                        osm_ctx.graph_fid_by_fid if osm_ctx is not None else None
                    )
                    for poly in prior_polys:
                        pfid = str(poly.get("fid") or "")
                        pmask = poly.get("mask")
                        if not pfid or pmask is None:
                            continue
                        if gmap:
                            pfid = gmap.get(pfid, pfid)
                        # UNIFIED COLOR CONTRACT: priors get M6 — the SAME
                        # extractor detections get — over the projected OSM
                        # footprint (accurate, not the old inflated CityNav
                        # hull), keyed on best-view pixels. Replaces the
                        # legacy single-word median so every mask, prior or
                        # detected, obeys one color contract.
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
