"""Tile-querying tools for the parse-first reasoning engine's resolver.

Each tool operates on the tile's spatial graph. Tools are read-only and
return JSON-serializable results.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any

from maps.sws.graph.model import ObjectGraph
from maps.sws.graph import build_object_graph

# ---------------------------------------------------------------------------
# OpenAI function-calling format conversion
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tool context
# ---------------------------------------------------------------------------

# Per-process graph cache — avoids redundant O(n²) graph builds for the same tile.
_GRAPH_CACHE: dict[str, ObjectGraph] = {}

_log = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Shared state holding tile data that all tools query against.

    ``grid_shape`` is the ``(H, W)`` frame. The prior ObjectGraph is
    built once per tile (cached by ``tile_name``) and supplies the
    feature index the resolver searches against. Feature index
    construction iterates the graph's ``objects_by_class`` — no closed
    vocabulary is assumed.
    """

    grid_shape: tuple[int, int]  # (H, W)
    features: dict = field(default_factory=dict)  # per-feature metadata
    # (name/tags/cells/cell_count/centroid/bbox/geo/orientation). Open-vocab:
    # any category key is allowed. Used by resolver tools and by the
    # executor's ``feature:<name>`` / ``features:ids=...`` source paths.
    tile_name: str | None = None  # stable key for cross-call caching
    _feature_index: dict = field(default_factory=dict, repr=False)
    graph: ObjectGraph | None = field(default=None, repr=False)
    # The one world-state graph: prebake seed plus every in-flight detection.
    # Nodes carry ``provenance`` ("prebaked" / "detected").
    robot_pose: object | None = None  # RobotPose; None when no pose available
    tile_origin_xy: tuple[float, float] = (0.0, 0.0)
    cell_size_m: float = 0.0  # required; caller passes the tile's true value

    def __post_init__(self) -> None:
        self._feature_index = _build_feature_index(self.features)
        if self.graph is not None:
            # Caller supplied a prior graph — use it as-is.
            pass
        elif self.tile_name and self.tile_name in _GRAPH_CACHE:
            self.graph = _GRAPH_CACHE[self.tile_name]
        else:
            # Rare fallback (every CityNav constructor passes ``graph=``):
            # build from features with the context's declared cell size —
            # ``cell_size_m`` is keyword-REQUIRED, omitting it was a latent
            # TypeError (2026-07-04 adversarial pass on the OSM prior bake).
            self.graph = build_object_graph(
                self.features, cell_size_m=self.cell_size_m
            )
            if self.tile_name:
                _GRAPH_CACHE[self.tile_name] = self.graph
        # Detected nodes already live in the one graph; fold them into the
        # feature index so text/id lookups and the compiler's feature-index
        # resolution reach them alongside prebake features.
        _augment_index_with_detected(self._feature_index, self.graph)


# ---------------------------------------------------------------------------
# Feature indexing
# ---------------------------------------------------------------------------


def _build_feature_index(features: dict) -> dict[str, dict]:
    """Build a stable ID -> feature dict from the features mapping.

    Iterates the per-class lists the caller supplied — no closed
    vocabulary. Non-list entries (``bbox``, ``georef``, ``_tile_name``,
    ``nlcd_zones``) are skipped.

    Keying mirrors the graph builder's id scheme so
    ``ctx._feature_index.get(graph_node_id)`` always resolves:
      * CityRefer features (``feat["cityref_id"]`` set) → ``cityref_<cityref_id>``
        (e.g. ``cityref_birmingham_block_0_3``). The cityref_id is
        already block-namespaced in the city aggregate; per-block
        bakes use the bare integer.
      * OSM-derived features (no ``cityref_id``) → ``{category}_{i}``
        (legacy by-position fallback).

    Without this matching, the graph carries cityref-id keys but the
    feature index uses by-position keys; the whitelist + searchable-text
    paths can't reach cityref features at all and the reasoner sees only
    bare ``{id, category, name, cell_count}`` for them. Note that the
    CityRefer ``descriptions`` list is deliberately stripped from the
    bake and the runtime tool surface (see the denylist in
    ``_extract_searchable_text``);
    "rich catalog text" here means OSM tags, names, object_type, etc.,
    not the per-object NL descriptions the benchmark instruction was
    drawn from.
    """
    index: dict[str, dict] = {}
    if not features:
        return index
    for category, items in features.items():
        if category in _NON_FEATURE_KEYS:
            continue
        if not isinstance(items, list):
            continue
        for i, feat in enumerate(items):
            if not isinstance(feat, dict):
                continue
            cityref_id = feat.get("cityref_id")
            if cityref_id:
                fid = f"cityref_{cityref_id}"
                feat.setdefault("source_tier", "cityref_prebake")
            else:
                fid = f"{category}_{i}"
                # Default OSM-derived features to the osm_prebake tier so the
                # feature index exposes source provenance downstream. SAM-derived
                # feature dicts already carry source_tier from the bake/upsert
                # path; setdefault preserves it.
                feat.setdefault("source_tier", "osm_prebake")
            index[fid] = {"category": category, "feature": feat}
    return index


_NON_FEATURE_KEYS: frozenset[str] = frozenset(
    {"bbox", "georef", "_tile_name", "nlcd_zones"}
)


def _augment_index_with_detected(
    index: dict[str, dict], graph: ObjectGraph | None
) -> None:
    """Fold runtime-PERCEIVED attributes from the live graph into the index.

    Two cases, both surfacing ONLY perception-derived appearance
    (``semantic_attributes``) — NEVER ``descriptions`` (the wall holds; the
    ranker's ``_SEARCHABLE_TEXT_DENYLIST`` keeps descriptions out regardless):

    1. **Detected nodes.** They live only in ``graph.nodes`` after
       ``upsert_detected`` — never in the OSM/CityRefer features dict. Without
       this, the executor's ``feature:NAME`` / ``features:ids=`` resolution and
       the feature-index text path cannot reach them. We inject a
       feature-shaped entry per detected node (centroid as a single cell so
       painters can mark it; ``semantic_attributes`` flattened so token search
       "gray house" matches).

    2. **Color-enriched priors.** A prebake node is ALREADY in the index from
       the static features dict, but that static entry is attribute-bare. When
       ``runtime.upsert_prior_percept`` has merged a PERCEIVED color onto the
       live node (the drone looked at the prior's footprint), fold that color
       onto the existing index entry's feature dict so a color query ranks the
       enriched prior — mirroring the detected path. Bare priors (empty
       ``semantic_attributes``) are left untouched.

    Snapshots ``graph.nodes.items()`` to a list before iterating so a
    sibling concept thread inserting under the upsert lock does not
    invalidate the iterator with ``dictionary changed size during
    iteration``.
    """
    if graph is None:
        return
    # ``graph.nodes.copy()`` is one C-level call (no GIL releases between
    # element copies), so it produces an atomic snapshot even when a
    # sibling concept thread is inserting under the upsert lock.
    # ``list(graph.nodes.items())`` still iterates the live view via
    # Python-level steps and can hit ``dictionary changed size``.
    for fid, node in graph.nodes.copy().items():
        if getattr(node, "provenance", "prebaked") != "detected":
            # Case 2: color-enriched PRIOR already in the index. Fold its
            # PERCEIVED attributes (color) onto the existing feature dict so
            # the ranker matches a color query; skip bare priors.
            sem = getattr(node, "semantic_attributes", None) or {}
            if not sem:
                continue
            existing = index.get(fid)
            if not existing:
                continue
            feat_existing = existing.get("feature")
            if not isinstance(feat_existing, dict):
                continue
            # COPY before merging: ``_build_feature_index`` stores REFERENCES to
            # the shared catalog feature dicts, so an in-place write would
            # leak this episode's perceived color into the static catalog
            # (and across episodes). Surface the color into the PER-CONTEXT
            # index entry only (the index dict is rebuilt fresh per ToolContext).
            merged = dict(feat_existing)
            wrote = False
            for key, counts in sem.items():
                if isinstance(counts, dict) and counts:
                    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
                    merged[key] = [v for v, _ in ranked]
                    wrote = True
            if wrote:
                existing["feature"] = merged
            continue
        if fid in index:
            continue
        # Surface the full sparse footprint when available so the painter
        # can paint a detected_N source as the actual SAM footprint, not
        # a single centroid cell. ``node.cells`` is the 2 m tile-grid
        # footprint (rasterized by the perception adapter); the painter
        # consumes 2 m cells directly.
        node_cells = getattr(node, "cells", None)
        if node_cells is not None and len(node_cells) > 0:
            cells = [[int(r), int(c)] for r, c in node_cells]
        else:
            cells = [[int(node.centroid[0]), int(node.centroid[1])]]
        tags = dict(node.tags) if node.tags else {}
        feat: dict[str, Any] = {
            "name": node.name,
            "tags": tags,
            "source_tier": tags.get("source_tier", "runtime_detection"),
            "matched_prebake_id": node.matched_prebake_id,
            "cells": cells,
            "centroid_row": int(node.centroid[0]),
            "centroid_col": int(node.centroid[1]),
            "cell_count": int(node.cell_count) or 1,
        }
        for key, counts in (node.semantic_attributes or {}).items():
            ranked = sorted(counts.items(), key=lambda kv: -kv[1])
            feat[key] = [v for v, _ in ranked]
        index[fid] = {"category": node.class_name, "feature": feat}


# ---------------------------------------------------------------------------
# Tool-output formatting
# ---------------------------------------------------------------------------
# Spatial primitives (centroid, bbox, orientation) come before nested arrays
# (relations, proximity) so the model reads the high-signal fields before
# wading into bucketed lists. Inner-bucket truncation is surfaced as a
# ``_truncated`` field; re-emitted entities carry ``_seen_at_turn`` so the
# reasoner can detect "I already queried this." JSON serializes compactly.
