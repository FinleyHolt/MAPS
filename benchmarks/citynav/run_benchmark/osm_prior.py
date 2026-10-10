"""Ortho OSM prior projection + color enrichment. Color is
ORDERING-only (CLAUDE.md): the enrichment feeds score_by, never a filter.
"""

from __future__ import annotations

from benchmarks.citynav.run_benchmark.config import (
    _GT_DIAGNOSTIC_OPT_OUT_ENV,
    _REALSTACK_PERCEPTION_TIER,
)
from dataclasses import dataclass, field
from pathlib import Path
from maps.perception.viz.color import color_from_mask_m6
from maps.search.analyze_scene.geom_compute import is_closed_chain
import json
import math
import time

import numpy as np
import os
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
    # SIDE STORE — perceived color for EVERY in-view OSM prior we see (named
    # AND unnamed), keyed by OSM fid (``cityref_<id>``). This is a downstream
    # capture ONLY: it is dumped to the ``flight/osm_percepts.json`` sidecar at
    # episode end and is NEVER added to the runtime ObjectGraph, never surfaced
    # via ``analyze_scene`` (its ``find`` op, which absorbed the old
    # ``search_features``), and never
    # enters the candidate / committable set — the headline ViSA named-only
    # contract (``benchmarks/citynav/PRIOR_CONTRACT.md``) stays intact. Only the
    # NAMED subset that ``graph_fid_by_fid`` maps to a real node is ALSO upserted
    # onto the graph (the committable, unchanged enrichment behavior). Each
    # entry: ``{name, class, world_xy, is_named, graph_fid, frames,
    # color:{value:count}, color_rgb:{value:count}}`` — color counts aggregate
    # across the stop's frames exactly like the graph path's ``{value:count}``.
    percept_by_fid: dict[str, dict] = field(default_factory=dict)


def _load_osm_prior_ctx(
    objects_json: str | Path, map_name: str, *, perception: str | None = None
) -> _OsmPriorCtx:
    """Load per-feature world contours for the perspective prior overlay.

    LEGACY / DIAGNOSTIC ONLY. Reads the CityRefer GT ``objects.json`` catalog
    (the parameter name is historical — this is GROUND-TRUTH geometry, not OSM).
    Under the real-stack pivot the prior-color enrichment context is re-sourced
    CityRefer-free from the baked OSM ``world_contour`` graph priors
    (:func:`_build_osm_prior_ctx_from_graph`), so this loader is HARD-REFUSED
    under the ``sam`` tier: reading objects.json GT geometry to steer
    which prior is color-enriched (and the color-sampling footprint) on the
    agent path is exactly the C1 violation SPEC §5c LEAK L1 closes. This choke-
    point guard makes an objects.json read IMPOSSIBLE under the pivot config no
    matter the call site. The single documented escape is the same one the
    runtime gate honors, ``SEARCH_ALLOW_GT_DIAGNOSTIC=1`` (a deliberate GT
    diagnostic probe — never a fleet run).
    """
    if (
        perception == _REALSTACK_PERCEPTION_TIER
        and os.environ.get(_GT_DIAGNOSTIC_OPT_OUT_ENV) != "1"
    ):
        raise RuntimeError(
            "C1 GT-ISOLATION VIOLATION (SPEC §5c LEAK L1): _load_osm_prior_ctx "
            "reads CityRefer objects.json GT contours, forbidden under the "
            f"{_REALSTACK_PERCEPTION_TIER!r} tier. Re-source prior enrichment "
            "from the baked OSM world_contour graph priors "
            "(_build_osm_prior_ctx_from_graph). Set "
            f"{_GT_DIAGNOSTIC_OPT_OUT_ENV}=1 ONLY for a deliberate diagnostic probe."
        )
    objs = json.loads(Path(os.path.expanduser(str(objects_json))).read_text())[map_name]
    contours: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for it in objs.values():
        cid = it.get("id")
        contour = it.get("contour")
        position = it.get("position")
        if cid is None or not contour or not position:
            continue
        fid = f"cityref_{cid}"
        z = float(position[2])
        contours[fid] = np.array([[float(x), float(y), z] for x, y in contour])
        meta[fid] = {
            "cityref_id": cid,
            "name": it.get("name"),
            "class_name": it.get("object_type") or "",
            "world_xy": (float(position[0]), float(position[1])),
        }
    return _OsmPriorCtx(contours_by_fid=contours, meta_by_fid=meta)


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


def _gt_nearest_prior_fid(
    graph, target_xy: tuple[float, float]
) -> tuple[str | None, float | None]:
    """GT-nearest prior fid for the tier-(c) oracle-set diagnostic
   : the contour-bearing prior node whose
    contour-centroid ``world_xy`` is nearest ``target_xy``. Same prior set
    and centroid as :func:`_build_osm_prior_ctx_from_graph` (and hence the
    ``flight/osm_percepts.json`` rows), computed from the graph because the
    sidecar does not exist at compile time. ``(None, None)`` when the graph
    carries no contour-bearing prior."""
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict) or not nodes:
        return None, None
    tx, ty = float(target_xy[0]), float(target_xy[1])
    best_fid: str | None = None
    best_d = math.inf
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
        if len(pts) < 3:
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        d = math.hypot(cx - tx, cy - ty)
        if d < best_d:
            best_fid, best_d = str(fid), d
    if best_fid is None:
        return None, None
    return best_fid, best_d


def _resolve_osm_prior_graph_fids(
    osm_ctx: "_OsmPriorCtx",
    graph,
    *,
    x_min: float,
    y_min: float,
    cell_size_m: float,
) -> None:
    """Map each OSM prior contour fid -> the runtime-graph node id it belongs to.

    The perspective prior-color enrichment projects OSM world contours
    (``osm_ctx``, keyed ``cityref_<id>`` from objects.json) onto the frame and
    upserts the perceived color onto a runtime-graph node. But the headline
    ViSA bake keys named priors ``<class>_<idx>`` — it strips ``cityref_id`` —
    so the OSM namespace never resolves against ``graph.nodes`` directly
    (measured: 0/1791 on birmingham_block_3; the graph holds 16 named roads +
    1 named building, none cityref-keyed). This fills ``osm_ctx.graph_fid_by_fid``
    once per episode by CLASS-GATED FOOTPRINT CONTAINMENT: an OSM prior maps to
    the prebaked node whose rasterized cells contain the prior's world centroid
    AND whose class is compatible (a ``Building`` prior -> a ``building`` node,
    a ``TrafficRoad`` prior -> a ``road`` node — never cross-class), the nearest
    such node when a cell is shared. Priors with no matching node stay unmapped,
    so the upsert no-ops honestly (e.g. cars / walls / street furniture, which
    are in nobody's runtime graph). Best-effort: any failure leaves the map empty
    (enrichment then no-ops, the pre-fix behavior) and never raises.
    """
    try:
        nodes = getattr(graph, "nodes", None)
        if not isinstance(nodes, dict) or cell_size_m <= 0:
            return
        # (row, col) cell -> list of (fid, class) for prebaked nodes.
        cell_owner: dict[tuple[int, int], list[tuple[str, str]]] = {}
        node_centroid_rc: dict[str, tuple[float, float]] = {}
        for fid, node in nodes.items():
            if getattr(node, "provenance", None) != "prebaked":
                continue
            cells = getattr(node, "cells", None)
            if cells is None:
                continue
            gclass = str(getattr(node, "class_name", "") or "")
            rsum = csum = ncell = 0
            for cell in cells:
                r, c = int(cell[0]), int(cell[1])
                cell_owner.setdefault((r, c), []).append((fid, gclass))
                rsum += r
                csum += c
                ncell += 1
            if ncell:
                node_centroid_rc[fid] = (rsum / ncell, csum / ncell)
        if not cell_owner:
            return

        def _compat(gclass: str, oclass: str) -> bool:
            g = gclass.strip().lower()
            o = oclass.strip().lower()
            if not g or not o:
                return False
            return g in o or o in g

        out: dict[str, str] = {}
        for fid, meta in osm_ctx.meta_by_fid.items():
            wxy = meta.get("world_xy")
            if not wxy:
                continue
            col = int((float(wxy[0]) - x_min) / cell_size_m)
            row = int((float(wxy[1]) - y_min) / cell_size_m)
            owners = cell_owner.get((row, col))
            if not owners:
                continue
            oclass = str(meta.get("class_name", "") or "")
            cands = [f for (f, g) in owners if _compat(g, oclass)]
            if not cands:
                continue
            if len(cands) == 1:
                out[fid] = cands[0]
                continue
            # Several same-class footprints share the cell: pick the nearest
            # node centroid (deterministic).
            best, bd = None, float("inf")
            for f in cands:
                crc = node_centroid_rc.get(f)
                if crc is None:
                    continue
                d = (crc[0] - row) ** 2 + (crc[1] - col) ** 2
                if d < bd:
                    bd, best = d, f
            if best is not None:
                out[fid] = best
        osm_ctx.graph_fid_by_fid = out
        log.info(
            "prior-enrichment: resolved %d/%d OSM priors to runtime-graph nodes "
            "(class-gated footprint containment)",
            len(out),
            len(osm_ctx.meta_by_fid),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "prior-enrichment: OSM->graph fid resolve failed (%s); perspective "
            "enrichment will no-op this episode",
            exc,
        )


def _merge_osm_percept(
    percept_by_fid: dict[str, dict],
    fid: str,
    meta: dict,
    attrs: dict,
    *,
    graph_fid: str | None,
) -> None:
    """Accumulate one in-view OSM prior's perceived color into the SIDE STORE.

    Called once per (frame, in-view OSM prior) in the enrichment hook for EVERY
    OSM prior — named AND unnamed — BEFORE any graph remap. The color counts
    aggregate as ``{value: count}`` across the stop's frames exactly like the
    runtime graph path (``maps.sws.graph.runtime.upsert_prior_percept``);
    ``frames`` counts the in-view sweeps for this fid. ``attrs`` may be empty
    (no dominant color this frame) — the fid is still recorded (frames bumps,
    metadata captured) so the side store reflects every OSM thing we saw.

    This writes ONLY to ``percept_by_fid`` (a plain dict on ``_OsmPriorCtx``
    dumped to the ``flight/osm_percepts.json`` sidecar). It never touches the
    runtime ObjectGraph and never reaches a reasoner tool surface — the named
    contract holds; the graph upsert is the caller's SEPARATE, named-only path.
    """
    entry = percept_by_fid.get(fid)
    if entry is None:
        wxy = meta.get("world_xy")
        name = meta.get("name")
        entry = {
            "name": name,
            "class": str(meta.get("class_name") or ""),
            "world_xy": (
                [float(wxy[0]), float(wxy[1])]
                if wxy is not None and len(wxy) >= 2
                else None
            ),
            "is_named": bool((name or "").strip()),
            "graph_fid": graph_fid,
            "frames": 0,
            "color": {},
            "color_rgb": {},
        }
        percept_by_fid[fid] = entry
    elif graph_fid and not entry.get("graph_fid"):
        entry["graph_fid"] = graph_fid
    entry["frames"] = int(entry.get("frames", 0)) + 1
    col = attrs.get("color")
    if col:
        bucket = entry["color"]
        bucket[col] = bucket.get(col, 0) + 1
    rgb = attrs.get("color_rgb")
    if rgb:
        bucket = entry["color_rgb"]
        bucket[rgb] = bucket.get(rgb, 0) + 1


def _write_osm_percepts_sidecar(
    workdir: str | None, percept_by_fid: dict[str, dict]
) -> int:
    """Dump the all-OSM perceived-color SIDE STORE to ``flight/osm_percepts.json``.

    Best-effort downstream capture (named + unnamed OSM priors, each with its
    aggregated perceived color). A write failure NEVER fails the flight — it is
    logged and swallowed. Adds a ``color_top`` / ``color_rgb_top`` convenience
    field (the highest-count value) alongside the full ``{value:count}`` maps.
    Returns the number of fids written (0 on no-op / failure).
    """
    if not workdir or not percept_by_fid:
        return 0
    try:
        out: dict[str, dict] = {}
        for fid, entry in percept_by_fid.items():
            color_map = dict(entry.get("color") or {})
            rgb_map = dict(entry.get("color_rgb") or {})
            rec = dict(entry)
            rec["color_top"] = (
                max(color_map, key=color_map.get) if color_map else None
            )
            rec["color_rgb_top"] = max(rgb_map, key=rgb_map.get) if rgb_map else None
            out[str(fid)] = rec
        flight_dir = Path(workdir) / "flight"
        flight_dir.mkdir(parents=True, exist_ok=True)
        target = flight_dir / "osm_percepts.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
        tmp.replace(target)
        return len(out)
    except Exception as exc:  # noqa: BLE001
        log.warning("prior-enrichment: osm_percepts.json sidecar write failed: %s", exc)
        return 0


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


def _build_all_osm_polys(
    *,
    rgb_shape: tuple[int, int],
    view_corners_xy,
    fids_to_render,
    step_idx: int = 0,
    target_fid: str = "",
    obs=None,
    osm_ctx: "_OsmPriorCtx | None" = None,
) -> tuple[list[dict] | None, list[dict] | None]:
    """Project ALL in-view OSM prior footprints (named + unnamed) to image
    masks for the downstream SIDE-STORE color capture.

    Read-only projection for the all-OSM ``percept_by_fid`` side store +
    ``flight/osm_percepts.json`` sidecar. It NEVER feeds the runtime graph or a
    reasoner surface; the named-only committable graph enrichment uses the
    separate :func:`_maybe_build_prior_polys` path. Original docstring follows.


    Returns ``(prior_polys, cityref_table)``. ``prior_polys`` carries
    per-feature masks the per-stop color enrichment reads pixels through (and
    that the prior overlay can draw);
    ``cityref_table`` mirrors the row shape minus the mask. Both are filtered to
    ``fids_to_render`` (OSM fids, ``cityref_<id>``).

    BOTH backends now source from the OSM contour context (``osm_ctx``), so the
    enrichment can color EVERY in-view OSM prior — named AND unnamed — not just
    the named ``base_features`` rows:

    - ORTHOGRAPHIC (``view_corners_xy`` set, e.g. ortho): projects each
      OSM world contour into the view square via
      :func:`build_prior_polys_ortho_from_contours` (the same ortho ``world ->
      pixel`` homography ``build_prior_polys_ortho`` uses, but over raw OSM
      contours instead of the named tile-grid ``cells``).
    - PERSPECTIVE (``view_corners_xy`` is ``None`` but ``obs.sensor`` +
      ``osm_ctx`` present): projects
      OSM world contours through the calibrated camera model via
      :func:`build_prior_polys_perspective`, applying the per-frame OSM↔SAM
      registration transform when present.

    Returns ``(None, None)`` when ``fids_to_render`` is empty or ``osm_ctx`` is
    absent (non-catalog map -> honest no-op). This is a read-only projection;
    keeping unnamed percepts out of the runtime graph is the caller's job (it
    upserts only fids ``graph_fid_by_fid`` maps to a named node).
    """
    if not fids_to_render or osm_ctx is None:
        return None, None
    try:
        # Orthographic path: project OSM world contours into the view square.
        from maps.perception.viewpoint.overlay import build_prior_polys_ortho_from_contours

        return build_prior_polys_ortho_from_contours(
            rgb_shape=rgb_shape,
            contours_world_by_fid=osm_ctx.contours_by_fid,
            feature_meta_by_fid=osm_ctx.meta_by_fid,
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


def _all_osm_fids_in_view(
    view_corners_xy, *, osm_ctx: "_OsmPriorCtx | None" = None, obs=None
) -> list[str]:
    """In-view OSM prior fids — BACKEND-AGNOSTIC, ALL OSM (named + unnamed).

    A cheap pre-filter so the per-frame footprint rasterization (via
    :func:`_maybe_build_prior_polys`) only runs over plausibly in-frame priors,
    not the whole city catalog. Both backends now cull over
    ``osm_ctx.meta_by_fid`` (EVERY thing in objects.json — named AND unnamed),
    so the color enrichment sees every in-view OSM prior, not just the named
    ``base_features`` rows. Returns OSM fids (``cityref_<id>``); the caller
    remaps the NAMED subset to a runtime-graph node for the committable upsert.

    ORTHOGRAPHIC backend (``view_corners_xy`` set, e.g. ortho): OSM
    prior centroids (``meta.world_xy``) whose xy lands within the view-square
    bbox (+10% margin). The EXACT in-frame test happens downstream in
    :func:`build_prior_polys_ortho_from_contours` (empty projected mask -> drop).

    PERSPECTIVE backend (``view_corners_xy`` is ``None`` but ``osm_ctx`` +
    ``obs.sensor`` present): OSM
    prior centroids within a generous ground radius of the camera pose. Only a
    coarse cull — :func:`build_prior_polys_perspective` drops any prior that
    projects behind / off the camera. The radius scales with altitude so a
    plausibly-visible prior is never pre-dropped; the projection does the real
    culling.

    Returns ``[]`` when ``osm_ctx`` is absent (non-catalog map) or the
    backend's pose/corner inputs are missing — enrichment then no-ops honestly
    rather than silently mis-firing."""
    if osm_ctx is None:
        return []
    if view_corners_xy:
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
        out: list[str] = []
        for fid, meta in (osm_ctx.meta_by_fid or {}).items():
            wxy = meta.get("world_xy")
            if not wxy:
                continue
            wx, wy = float(wxy[0]), float(wxy[1])
            if x0 <= wx <= x1 and y0 <= wy <= y1:
                out.append(str(fid))
        return out

    # Perspective backend: coarse pose-radius cull over the OSM prior contours.
    if obs is None or getattr(obs, "sensor", None) is None:
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
    """Pure per-frame OSM-enrich BUILD: in-view fids, poly rasterization,
    and color extraction for both enrich passes — everything except the
    graph/side-store APPLY, which stays serial in the flight body. Reads
    only the frame (obs.rgb, view_corners) and episode-static context
    (assets.base_features, osm_ctx meta/contours/graph_fid_by_fid). A
    failure is carried (not raised) so the apply replays the serial
    rate-limited warn + fail-count semantics; items completed before the
    failure are returned, matching the serial partial-apply behavior.
    Returns {"prior": [(pfid, pattrs, ppx)...], "prior_exc",
             "osm": [(osm_fid, sattrs)...], "osm_exc"}.
    """
    out: dict = {"prior": [], "prior_exc": None, "osm": [], "osm_exc": None}
    # Where this frame's enrichment went. The whole build is one mark in the
    # stop's phase line; these say which of its two passes spent it.
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
    if osm_ctx is not None:
        try:
            _m = _c()
            osm_fids = _all_osm_fids_in_view(view_corners, osm_ctx=osm_ctx, obs=obs)
            _t["o_fids"] = _c() - _m
            _t["o_n"] = float(len(osm_fids or []))
            _m = _c()
            if rgb_arr.ndim == 3 and osm_fids:
                osm_polys, _ = _build_all_osm_polys(
                    rgb_shape=(int(rgb_arr.shape[0]), int(rgb_arr.shape[1])),
                    view_corners_xy=view_corners,
                    fids_to_render=osm_fids,
                    step_idx=step_idx,
                    obs=obs,
                    osm_ctx=osm_ctx,
                )
                _t["o_polys"] = _c() - _m
                _m = _c()
                if osm_polys:
                    from maps.perception.sam.detect_client import (
                        _set_color_attr as _sca,
                    )

                    for poly in osm_polys:
                        osm_fid = str(poly.get("fid") or "")
                        pmask = poly.get("mask")
                        if not osm_fid or pmask is None:
                            continue
                        sattrs: dict = {}
                        _sca(sattrs, rgb_arr, pmask)
                        out["osm"].append((osm_fid, sattrs))
                _t["o_color"] = _c() - _m
        except Exception as exc:  # noqa: BLE001 - carried to the apply
            out["osm_exc"] = exc
    out["phase_s"] = _t
    return out
