"""Runtime graph mutation — track_id-keyed detected-node upsert.

A detection folds onto its existing ``detected_<N>`` node by EXACT
``track_id`` (``_find_track``; ByteTrack is the sole re-id signal — no IoU,
voxel, or centroid fallback) or mints a new one. The only tag separating a
runtime node from a prebake node is ``MapObject.provenance="detected"`` plus
its per-observation counters (``first_seen_t``, ``last_seen_t``,
``observation_count``, ``confidence``, ``semantic_attributes`` running
aggregates).

The graph is nodes-only: it carries no spatial edges. The reasoner reasons
over the ``analyze_scene`` surface, which grounds via ``node.centroid``, and
every spatial relation is computed geometrically from node xy/cells by the
geom_compute ops.
``upsert_detected_centroid_only`` is the off-canonical insurance mint.
"""

from __future__ import annotations

import logging
import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)

from maps.sws.graph.model import MapObject, ObjectGraph


# Cap on embeddings stored per prebake node; bounds the serialized
# prebake_evidence.json size and the in-memory aggregate at ~16 * embedding_dim
# floats per node. SAM 3.1 detections rarely contribute embeddings today, but
# the cap keeps future per-frame DINO-style embeddings from snowballing.
_PREBAKE_EVIDENCE_EMBEDDING_CAP = 16


@dataclass
class TrackContribution:
    """Per-track ledger of one track's accumulated fold onto a prebake.

    Every entry in :attr:`PrebakeEvidence.per_track` is the cumulative
    contribution of one ``track_id`` to the parent prebake across every
    frame the matcher attributed it to that prebake. The aggregate
    counters on :class:`PrebakeEvidence` are the sum of all
    TrackContribution entries; subtracting one entry from the aggregate
    is what makes a revoke exact.
    """

    confidence_sum_added: float = 0.0
    observation_count_added: int = 0
    attribute_increments: dict[str, dict[str, int]] = field(default_factory=dict)
    evidence_refs_added: list[str] = field(default_factory=list)
    embeddings_added_count: int = 0
    first_added_t: float | None = None
    last_added_t: float | None = None


@dataclass
class PrebakeEvidence:
    """SAM evidence aggregated onto a prebake node.

    The prebake's geometry (cells, bbox, centroid, altitude_m, height_m,
    contour) is authoritative and immutable; this sidecar carries the
    per-observation signal SAM produces (confidence, attributes,
    embeddings, track ids) without mutating the MapObject.

    Every fold is also recorded in :attr:`per_track` so a lazy-revoke
    can subtract the exact contribution of one track from the aggregate
    counters when the lifecycle state machine in
    ``maps.search.search_loop`` switches or detaches a track's attribution.
    """

    confidence_sum: float = 0.0
    observation_count: int = 0
    attributes: dict[str, dict[str, int]] = field(default_factory=dict)
    embeddings: list[list[float]] = field(default_factory=list)
    track_ids: set[str] = field(default_factory=set)
    evidence_refs: list[str] = field(default_factory=list)
    first_seen_t: float | None = None
    last_seen_t: float | None = None
    per_track: dict[str, TrackContribution] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        n = self.observation_count
        return self.confidence_sum / n if n > 0 else 0.0


def get_prebake_evidence(graph: ObjectGraph) -> dict[str, PrebakeEvidence]:
    """Return the per-graph prebake-evidence sidecar, creating it lazily."""
    return graph.__dict__.setdefault("_prebake_evidence", {})


def revoke_track_contribution(
    graph: ObjectGraph,
    track_id: str,
    *,
    only_fids: Iterable[str] | None = None,
) -> list[str]:
    """Subtract one track's accumulated fold from every prebake it touched.

    Walks the per-graph prebake-evidence bucket, pops
    :class:`TrackContribution` ledger entries for ``track_id``, and
    subtracts each ledger entry from the parent aggregate counters. The
    track is removed from :attr:`PrebakeEvidence.track_ids`, the
    evidence refs the track added are dropped, and the trailing ``N``
    embeddings (where ``N`` is :attr:`TrackContribution.embeddings_added_count`)
    are popped from the embeddings list. Empty attribute buckets are
    pruned.

    Returns the list of prebake fids the track was revoked from. When
    ``only_fids`` is provided, scoping is limited to those fids; other
    prebakes the track may have folded onto are left untouched.
    """
    bucket = get_prebake_evidence(graph)
    if not bucket:
        return []
    tid = str(track_id)
    targets = set(only_fids) if only_fids is not None else None
    revoked: list[str] = []
    for fid, ev in bucket.items():
        if targets is not None and fid not in targets:
            continue
        contrib = ev.per_track.pop(tid, None)
        if contrib is None:
            continue
        ev.observation_count = max(
            0, ev.observation_count - contrib.observation_count_added
        )
        ev.confidence_sum = max(0.0, ev.confidence_sum - contrib.confidence_sum_added)
        ev.track_ids.discard(tid)
        for ref in contrib.evidence_refs_added:
            try:
                ev.evidence_refs.remove(ref)
            except ValueError:
                pass
        if contrib.embeddings_added_count > 0:
            # Embeddings are appended in fold order; pop the trailing N as
            # the cheapest revocation. The set is unordered for downstream
            # consumers, so LIFO removal is safe.
            keep = max(0, len(ev.embeddings) - contrib.embeddings_added_count)
            del ev.embeddings[keep:]
        for key, value_counts in list(contrib.attribute_increments.items()):
            agg = ev.attributes.get(key)
            if agg is None:
                continue
            for value, count in value_counts.items():
                agg[value] = max(0, agg.get(value, 0) - int(count))
                if agg[value] == 0:
                    agg.pop(value, None)
            if not agg:
                ev.attributes.pop(key, None)
        revoked.append(fid)
    return revoked


# The four M6 color attrs (+ color_rgb) that travel together as ONE
# best-view read. Overwritten atomically as a set so a partial stale mix never lingers.
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
    """Best-view color contract — uniform for detected AND prebaked-prior masks.

    REPLACE the node's color buckets with ``color_attrs`` iff this frame's in-view
    mask pixel count ``px`` strictly exceeds the node's stored ``best_view_px`` — so
    the perceived color is read from the single frame in which the object subtended
    the most pixels (closest / least-occluded), not frozen at first sight and not a
    modal vote dominated by far/small views. All color keys move together (one M6
    read); an empty/absent value clears its key so a worse-view remnant can't linger.
    Returns True when it replaced (a strictly better view landed).

    LOCKSTEP INVARIANT: callers MUST pass ``px`` (>0) in lockstep with a real M6
    color read — because a larger view CLEARS any key absent from ``color_attrs``, a
    caller that ever bumped ``px`` without the accompanying color would wipe a good
    earlier read. Every live site upholds this: ``color_px`` is stamped only inside
    ``if mix:`` (M6 succeeded), so px>0 ⟺ color attrs present."""
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
    """Merge a PERCEIVED attribute set onto a prebake node from the pixels.

    ``attrs`` is the M6 color read over the prior's projected OSM footprint —
    the SAME extractor a SAM detection gets (``perception.color``), e.g.
    ``{"color": "blue 0.7 / black 0.2", "color_conf": "0.83", "color_shade":
    "mid", "color_flag": ""}``. Color keys obey the best-view contract
    (:func:`apply_best_view_color`, keyed on ``px`` = in-view mask pixels):
    they REPLACE, not accumulate, so the read from the frame where the prior
    subtended the most pixels wins — identical to detected nodes. Any other
    keys still accumulate as ``{value: count}``. ``observation_count`` /
    ``first_seen_t`` / ``last_seen_t`` bump on every non-empty sighting so the
    enriched prior surfaces as an OBSERVED, color-bearing candidate
    (``collect_observed_target_candidates``).

    This DELIBERATELY overturns the "prebake nodes stay attribute-bare /
    never mutated" convention — but ONLY for PERCEIVED attributes derived
    from pixels. The descriptions wall stays intact: nothing here ever reads
    ``descriptions`` / ``name`` / the catalog. The node's authoritative
    GEOMETRY (cells, bbox, centroid, altitude) is never touched.

    Returns True when a prebake node was enriched. Robust by construction:
    a missing fid, a non-prebake node, or empty ``attrs`` is a no-op that
    returns False; it never raises (so a per-frame enrichment sweep can call
    it best-effort).
    """
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict) or not attrs:
        return False
    node = nodes.get(str(fid))
    if node is None or getattr(node, "provenance", None) != "prebaked":
        return False
    color_attrs = {k: attrs[k] for k in attrs if k in _BEST_VIEW_COLOR_KEYS}
    other_attrs = {k: attrs[k] for k in attrs if k not in _BEST_VIEW_COLOR_KEYS}
    # A prior's color obeys the SAME best-view contract as a detection: the read
    # from the LARGEST in-view footprint (px) wins, REPLACING — never accumulating a
    # modal vote that the many far/small views would dominate.
    touched = apply_best_view_color(node, color_attrs, px)
    # A color SIGHTING (even one that did not beat the stored best view) still
    # surfaces the prior as OBSERVED, so bump on any non-empty color read.
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
    """Return the next ``detected_<idx>`` id for ``graph``.

    Uses a monotonic counter cached on the graph instead of rescanning
    the full node dict every call. The old O(N) scan was the dominant
    Python cost in city-scale ingest: at K_fresh=4000 against n0=30k it
    burned ~18 s per concept doing nothing but building sets of existing
    ids. The counter is initialized lazily from the current graph state
    (so existing IDs from disk are respected) and bumped on every issue.
    """
    counter = graph.__dict__.get("_next_detected_idx")
    if counter is None:
        # Lazy init: scan once to find the highest existing ``detected_<n>``
        # id, then count from there. Subsequent calls skip the scan.
        max_idx = -1
        for nid in graph.nodes:
            if nid.startswith("detected_"):
                try:
                    max_idx = max(max_idx, int(nid[len("detected_") :]))
                except ValueError:
                    continue
        counter = max_idx + 1
    fid = f"detected_{counter}"
    # Defensive against externally-injected ids that might collide.
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
    evidence_id: str | None = None,
) -> str:
    """Off-canonical insurance: mint a bare detected node with NO mask/dedup
    machinery. Returns the new ``detected_<N>`` id.

    The canonical ``ortho`` path stamps a ``track_id`` on every
    detection (the world tracker) and routes through
    :func:`upsert_tracked_detected_batch`; this handles only a track_id-less
    detection, which the world-state cleanout reduced to a centroid-only node
    (``cells``/``embedding`` are already None from the producer). ByteTrack is
    the sole re-id signal, so a detection without a ``track_id`` mints its own
    node — there is deliberately no IoU/centroid/embedding dedup fallback.
    """
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
    if evidence_id is not None and evidence_id not in node.evidence_refs:
        node.evidence_refs.append(evidence_id)
    graph.nodes[fid] = node
    if graph.objects_by_class:
        graph.objects_by_class.setdefault(node.class_name, {})[fid] = node
    if step_index is not None:
        graph.step_index = step_index
    graph.tau = now
    return fid




# ---------------------------------------------------------------------------
# Tracked upsert (mask-aware, byte-tracked across frames)
# ---------------------------------------------------------------------------


def _stamp_dedup_tag(
    node: MapObject,
    *,
    status: str,
    track_info: dict,
    prebake_info: dict | None,
) -> None:
    """Write the dedup record onto ``node.tags['dedup']`` as JSON. The
    engine layer reads this back into ``DetectedObject.metadata['dedup']``
    so downstream audit / dashboard consumers see one canonical schema."""
    rec = _dedup_record(
        status=status,
        track_info=track_info,
        prebake_info=prebake_info,
    )
    node.tags["dedup"] = json.dumps(rec)


def _dedup_record(
    *,
    status: str,
    track_info: dict,
    prebake_info: dict | None,
) -> dict:
    """Assemble the production dedup record stamped on the detected node.

    Stamped onto ``MapObject.tags['dedup']`` as JSON so the engine /
    dashboard / audit tools can inspect every match decision after the
    fact. ``status`` is one of ``confirmed`` (prebake or detected match),
    ``new`` (no candidate above threshold), or ``confirmed_track`` (exact
    track_id reuse).
    """
    rec: dict = {"status": status}
    if prebake_info:
        rec["match_score"] = float(prebake_info.get("match_score", 0.0))
        if "candidates" in prebake_info:
            rec["candidates"] = prebake_info["candidates"]
        if "top3" in prebake_info:
            # Lifecycle state machine in maps.search.search_loop reads this to
            # drive strengthen / hold / switch / detach decisions per track.
            rec["top3"] = prebake_info["top3"]
        if "all_attribution_fids" in prebake_info:
            rec["all_attribution_fids"] = prebake_info["all_attribution_fids"]
        if status == "confirmed":
            match: dict = {}
            for key in (
                "fid",
                "prebake_class",
                "inter",
                "sam_voxel_count",
                "prebake_voxel_count",
                "frac_in_sam",
                "frac_of_prebake",
                "reverse_coverage",
                "z_extent_sam",
                "z_extent_prebake",
            ):
                if key in prebake_info:
                    match[key] = prebake_info[key]
            rec["match"] = match
        if "best_near_miss" in prebake_info and prebake_info["best_near_miss"]:
            rec["best_near_miss"] = prebake_info["best_near_miss"]
    elif track_info:
        if "primary" in track_info:
            rec["primary"] = track_info["primary"]
        rec["match_score"] = float(track_info.get("match_score", 0.0))
        if "candidates" in track_info:
            rec["candidates"] = track_info["candidates"]
        if track_info.get("best_near_miss"):
            rec["best_near_miss"] = track_info["best_near_miss"]
    return rec


def _cells_from_mask(hm_rows: np.ndarray, hm_cols: np.ndarray) -> np.ndarray:
    """The ``(N, 2) int32`` (row, col) footprint of a detection's projected hard
    mask — the SAME lean sparse representation prebake nodes carry (the builder
    sets ``cells=arr``), so a detected car and a prebake building expose ONE
    footprint format. Lean by construction: a car footprint is a handful of 2 m
    cells (tens of bytes), never the dense per-node raster the world-state
    cleanout removed."""
    return np.stack((hm_rows, hm_cols), axis=1).astype(np.int32)


def _stamp_cells(
    node: MapObject, hm_rows: np.ndarray, hm_cols: np.ndarray, cell_size_m: float
) -> None:
    """Set a detected node's footprint AND the geometry derived from it.

    THE single place a detected node's ``cells`` are written, so the derived
    centerline family cannot be forgotten at one of the three sites that write
    them. It matters because a graph that is never SAVED is never LOADED, and
    ``io.derive_centerlines`` runs inside ``load_graph``: the world-state ingest
    SERVER path returns its graph through that round trip on every call and the
    in-process path never does. Without this, the same detections produced
    centerline-carrying nodes through the server and centerline-less nodes
    in-process — and the geom ops resolve a centerline-carrying fid as a LINE
    reference and a centerline-less one as a point
    (``analyze_scene/geom_program.py`` ``_cl`` / ``_axis_q``), so the difference
    reached answers. Calling the loader's own function, rather than a second
    copy of the reduction, is what makes the two agree by construction instead
    of by review."""
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
    frame_idx: int,
    now: float | None = None,
    evidence_id: str | None = None,
) -> str:
    """Upsert a tracked detection from its projected ground cells.

    ``rows`` / ``cols`` are the deduplicated occupied cells of the
    detection's footprint on the tile grid, in row-major order
    (``TileProjectedMask.rows`` / ``.cols``).

    First sighting of ``track_id`` allocates a new ``detected_*`` node.

    Repeat sightings look up the existing node by ``track_id`` (ByteTrack
    is the sole dedup) and replace the node attrs in place.

    ``base_node`` carries the class/tags/label/embedding for fresh
    inserts; its centroid and footprint are overwritten from the cells so
    the caller doesn't have to pre-populate them.
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
        # Empty observation — nothing to upsert.
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
        # ByteTrack-only / uniform-node model: every detection mints its own
        # ``detected_<N>`` node — it never folds onto a prebake. Prebakes and
        # detections coexist as distinct nodes, distinguished only by the
        # ``provenance`` tag; there is no detection→prebake ``matched_prebake_id``
        # link and no prebake-evidence fold.
        prebake_match: str | None = None
        prebake_match_info: dict | None = None

        fid = _next_detected_id(graph)
        base_node.id = fid
        base_node.track_id = track_id
        base_node.provenance = "detected"
        base_node.centroid = centroid_rc
        base_node.bbox = bbox
        base_node.cell_count = cell_count
        # Per-detection mask tracking: store the projected
        # footprint cells in the SAME format prebake nodes use, so build_surface
        # attaches a boundary + shape features to this detection uniformly.
        _stamp_cells(base_node, hm_rows, hm_cols, graph.cell_size_m)
        base_node.confidence = float(confidence)
        base_node.observation_count = 1
        base_node.first_seen_t = now
        base_node.last_seen_t = now
        if evidence_id is not None and evidence_id not in base_node.evidence_refs:
            base_node.evidence_refs.append(evidence_id)
        base_node.matched_prebake_id = prebake_match
        _stamp_dedup_tag(
            base_node,
            status="new",
            track_info={},
            prebake_info=prebake_match_info,
        )

        graph.nodes[fid] = base_node
        if graph.objects_by_class:
            graph.objects_by_class.setdefault(base_node.class_name, {})[fid] = base_node
        return fid

    # Repeat sighting: update the existing node in place.
    node = graph.nodes[existing_fid]
    # Re-stamp the current frame's track_id and projected cells so the
    # next exact-track lookup sees fresh state. The world tracker's
    # ByteTrack id is the sole re-identification signal, so pinning the
    # latest id is what keeps the exact-match path landing here.
    node.track_id = track_id
    node.centroid = centroid_rc
    node.bbox = bbox
    node.cell_count = cell_count
    # Per-detection mask tracking: re-stamp the current
    # frame's projected footprint so the node's cells track the latest
    # observation (centroid/bbox above are re-derived from the same cells).
    _stamp_cells(node, hm_rows, hm_cols, graph.cell_size_m)
    # WORLD_CONTOUR re-stamp (same spirit as the cells re-stamp above): when
    # the incoming detection carries a fine world-meter footprint polygon,
    # the latest observation wins; a contour-less repeat never clears it.
    incoming_contour = getattr(base_node, "world_contour", None)
    if incoming_contour:
        node.world_contour = incoming_contour
    incoming_wxy = getattr(base_node, "world_xy", None)
    if incoming_wxy is not None:
        node.world_xy = (float(incoming_wxy[0]), float(incoming_wxy[1]))
    # UNIFIED COLOR CONTRACT: replace color with this frame's M6 read only when this
    # view is larger than the best seen so far. base_node.best_view_px carries the new
    # frame's mask pixel count (stamped in search_loop._detection_to_mapobject); its color
    # buckets carry the new M6 read. Frozen-at-first-sight color is gone.
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
    if evidence_id is not None and evidence_id not in node.evidence_refs:
        node.evidence_refs.append(evidence_id)
    _stamp_dedup_tag(
        node,
        status="confirmed_track",
        track_info={"primary": "track_id", "match_score": 1.0},
        prebake_info=None,
    )

    return existing_fid


def _find_track(graph: ObjectGraph, track_id: str) -> str | None:
    """Locate the detected node for ``track_id`` — exact match only.

    The world-space tracker (ByteTrack/UCMCTrack-core) assigns a stable,
    class-gated ``track_id`` per physical object, so the exact ``track_id``
    match is the SOLE re-identification signal: same track_id → same node,
    different track_id → a distinct node. ByteTrack is the only dedup — there
    is no geometry (mask-IoU) fallback, and no prebake fold.
    """
    for fid, node in graph.nodes.items():
        if node.provenance == "detected" and node.track_id == track_id:
            return fid
    return None


def upsert_tracked_detected_batch(
    graph: ObjectGraph,
    replaces: list[dict],
) -> list[str]:
    """Apply :func:`upsert_tracked_detected` to each replace dict in order;
    the node ids in input order."""
    return [upsert_tracked_detected(graph, **r) for r in replaces]
