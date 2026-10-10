"""Feature graph data structures for spatial relationship modeling.

Core types:
  MapObject  — a spatial feature with geometry and metadata
  ObjectGraph — dict-backed spatial graph with per-class object sets O_n^{(c)}
  GridDomain   — raster grid X = {1..H} × {1..W} per heater_notes.tex
  WorldState   — W_n snapshot carrying (I, V_n, {M_n^c, O_n^c}_c) pieces
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GridDomain:
    """Raster grid X = {1..H} × {1..W} with cell resolution.

    Always 0-indexed in numpy arrays. H = row count (north→south),
    W = column count (west→east). cell_size_m is metric resolution.
    """

    height: int
    width: int
    cell_size_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


@dataclass
class MapObject:
    """A feature in the spatial graph.

    Its mask is materialized on demand from ``cells``/``boundary``; its
    centroid is ``centroid`` (row, col).

    Runtime fields (``provenance="detected"``) carry per-observation metadata
    so the orchestrator can trust-weight and age detected nodes independently
    of prebaked OSM / CityRefer nodes.
    """

    id: str  # "roads_5", "water_0", "detected_0"
    class_name: str  # prebake class (e.g. "roads", "water", "buildings") or, for
    #                  a runtime detection, its fine SAM concept label ("white car")
    name: str | None
    tags: dict[str, str]
    centroid: tuple[int, int]  # μ_o, (row, col)
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    cell_count: int
    orientation: float | None  # Degrees [0, 180), 0=N-S, 90=E-W
    orientation_confidence: float = 0.0  # 0=round/ambiguous, 1=clearly linear
    geo: tuple[float, float] | None = None  # (lat, lon) WGS84 centroid
    boundary: list[list[float]] | None = None  # [[row, col, nr, nc], ...]
    # Natural-language descriptions of this object (CityRefer ships ~4-6 per
    # object, e.g. "this is a building with a gray roof and red walls...").
    # Prebaked datasets without per-object NL leave this empty.
    #
    # GROUND-TRUTH LABEL LEAK: do NOT surface this field to any runtime LLM
    # agent. That is the single persistent MAPS Reasoner across every round
    # it runs — round-0 grounding, the per-stop reasoner rounds, and
    # the deterministic end-of-tour commit all share the one reasoner context
    # and inherit this constraint. The per-object descriptions are the
    # superset CityRefer drew the operator instruction from; the per-episode
    # instruction is one item of this list. An agent that can read both can
    # string-match its instruction back to the matching catalog entry and
    # recover ground-truth target id without ever looking at the world. The
    # CityNav benchmark protocol provides only the single per-episode
    # instruction (see benchmarks/citynav/bake/city.py for
    # the bake-side rule and maps/search/analyze_scene/geom_program.py
    # for the find-op surface wall — ``_anchor_search_text`` /
    # ``_SEARCH_TAG_DENYLIST``; the ``find`` op absorbed the old
    # ``search_features`` / ``get_feature_context``).
    # New tools, new dashboards, and new evaluators all inherit this
    # constraint — if a feature dict is going to an LLM, descriptions stays
    # behind. The field stays on MapObject so non-LLM consumers (audit,
    # ground-truth scoring, dataset analysis) can still read it.
    descriptions: list[str] = field(default_factory=list)
    provenance: str = "prebaked"  # "prebaked" (OSM / CityRefer) | "detected"
    # Prebake feature id this detection corresponds to (class-compatible
    # prebake whose footprint overlaps the detection mask above threshold
    # at first insertion). Informational only: the prebake node's
    # authoritative GEOMETRY is never mutated (perceived attributes ARE now
    # merged onto a prior via ``runtime.upsert_prior_percept`` — render-and-
    # look color, never catalog fields), so this link is what downstream
    # audit tools use to tell "new detection" from "detection that confirms
    # the map".
    matched_prebake_id: str | None = None
    first_seen_t: float | None = None  # wall-clock of first observation
    last_seen_t: float | None = None
    observation_count: int = 0
    # Best-view color provenance: the greatest in-view mask pixel count from which
    # this node's perceived color was extracted. The runtime upsert REPLACES the
    # color attrs (color/color_conf/color_shade/color_flag) only when a later frame's
    # mask beats this, so color comes from the single best (largest, least-occluded)
    # view of the object — the SAME contract for detected and prebaked-prior masks.
    # 0 = no color view yet. (Overwrite, not the {value:count} accumulate the other
    # semantic_attributes use — a modal vote would keep the many far/small views over
    # the one close read.)
    best_view_px: int = 0
    confidence: float = 1.0  # aggregated detection confidence [0,1]
    # P(this node is the instruction's referent | observations). Default 0.0
    # so unscored nodes bias the planner zero (the visibility multiplier is
    # 1 + lambda * field). Populated at runtime by the reasoner's scoring.
    is_target_confidence: float = 0.0
    embedding: list[float] | None = None  # optional vector for dedup / re-id
    semantic_attributes: dict[str, dict[str, int]] = field(default_factory=dict)
    # Tracker-assigned stable identity across frames. None for prebake nodes
    # and for untracked point detections; set by the world tracker (or any
    # tracking adapter that writes ``DetectedObject.metadata["track_id"]``)
    # when a detection is associated with a persistent masklet.
    track_id: str | None = None
    # Open-vocab attributes observed on a node: {attribute_key: {value: count}}.
    # Keys are not whitelisted (color, material, text_span, vlm_descriptor, …);
    # counts accumulate across upserts so repeated color readings reinforce each other.
    # PREBAKE nodes are enriched too — but ONLY with PERCEIVED attributes (render-and-look color over
    # the prior's projected footprint, via ``runtime.upsert_prior_percept``), NEVER
    # from descriptions/name/the catalog (the descriptions wall is intact).
    # Rasterized tile-grid footprint cells, shape ``(N, 2) int32`` (row, col) —
    # ONE lean format across every node type. PREBAKE nodes carry their footprint
    # cells (``builder.build_object_graph`` sets ``cells=arr``); DETECTED nodes
    # carry their per-observation projected footprint too
    # (``runtime._cells_from_mask`` off the authoritative ``grid_mask``, re-stamped
    # each frame; the SAM detection-mask tracking). Both are round-tripped by
    # ``io`` and consumed uniformly: build_surface realises the EXACT
    # multi-loop boundary (the relation/gap substrate — every fragment and
    # hole, no decimation) + the mask-derived shape columns (footprint_area_m2
    # / aspect_ratio) for detected and prebake nodes alike, and
    # color-enrichment containment / the viewpoint / painter paths read cells
    # directly. This is the sparse cell list ONLY: no dense per-node raster
    # stack is kept, for memory (re-ID is exact track_id only, never
    # mask-overlap).
    # A centroid-only detection mint (``upsert_detected_centroid_only``, no
    # grid_mask) legitimately leaves this ``None`` and gets no footprint columns.
    cells: np.ndarray | None = None
    # WORLD_CONTOUR substrate: the feature's EXACT vertex chain in WORLD
    # METERS — ``[[x, y], ...]`` at 2 dp — replacing the 2 m-cell-quantized
    # boundary wherever a fine source exists. ONE representation for every
    # geometry the map or the sensor gives: a single point, an open way, or a
    # closed ring, which STATES its closure by returning to its first vertex
    # (``geom_compute.is_closed_chain``). Nothing reading this field asks what
    # kind of feature it belongs to; distance is nearest point to nearest
    # point on whatever vertices are here. A MEASURED
    # contour is decimated by its producer (the vertex budget of a pixel-mask
    # outline is a rendering choice); a MAP contour is not, because it IS the
    # footprint rather than an observation of one — a 16-vertex uniform stride
    # cost one Cal City ring 99% of its area, and footprint_area_m2 is a
    # column the model scores and filters on. Sources:
    # gt_perfect stamps the exact CityRefer contour (tightness-gated — loose
    # annotation hulls are withheld); the real-SAM ortho path derives it from
    # the native-res pixel mask via the view-square affine
    # (``perception.geolocate.world_contour_from_pixel_mask``). ADDITIVE next
    # to ``cells`` (which stay for coverage ops): build_surface prefers this
    # for the row ``mask`` + computes the shape scalars from it regardless
    # of cell count (a 2-cell car with a contour gets a real orientation
    # instead of the 0/90-quantized garbage the S2 gate suppresses).
    # Re-stamped on repeat sighting when the incoming detection carries one
    # (same spirit as the ``cells`` re-stamp); never cleared by a
    # contour-less repeat.
    world_contour: list[list[float]] | None = None
    # Continuous world-meter position from the perceiving side (the tracker's
    # sub-cell estimate) — ADDITIVE next to the int cell centroid, same
    # pattern as world_contour: the 2 m cell grid quantizes arrangement
    # structure away (a 2.4 m in-row car spacing is unresolvable at 2 m
    # cells — measured: it erased a real lot's lattice), so geometry readers
    # prefer this when present and fall back to the cell-center.
    world_xy: tuple[float, float] | None = None
    # --- Raw per-node spatial geometry (Tier-A precompute; mask-derived) ---
    # Computed generically for EVERY node from its mask cells, in the SAME
    # grid (row, col) frame as ``centroid``/``cells``/``bbox`` — NOT world xy.
    # The builder has no tile origin, and the row→world mapping + cardinal
    # sign are frame-sensitive (grid row 0 is the SOUTH edge, +row = +north;
    # see the frame-convention geometry in maps.search geom_compute). So we bake only
    # raw, frame-NEUTRAL geometry and let the tool boundary do the single,
    # already-correct world conversion. Frame-DEPENDENT quantities (compass
    # bearing, signed left/right) are NEVER baked: tools derive them from this
    # raw geometry at query time. Thresholded interpretation (is_near / Nth /
    # left-of) is a Tier-B model decision and is never stored here.
    #
    # ``centerline`` is the oriented principal-axis polyline of the mask:
    # ordered (row, col) vertices sampled at the grid resolution along the PCA
    # axis (lateral mask-midpoint per step). For a compact mask it is a short
    # stub with low ``orientation_confidence``; for an elongated mask (a road,
    # a row of buildings) it is the object's centerline. The model grounds an
    # axis onto whichever node the instruction names — roads are never special.
    centerline: list[list[float]] | None = None  # [[row, col], ...] grid frame
    centerline_arc_m: list[float] | None = None  # cumulative meters per vertex
    axis_length_m: float = 0.0  # total centerline arc length, meters
    # Max perpendicular distance (meters) of a mask cell to the straight PCA
    # line: a curvature/fit beacon (≈ half-width for a straight mask, inflated
    # by curvature). The visualizer is the authoritative check; this is a hint.
    axis_residual_m: float = 0.0
    centerline_method: str = ""  # "pca_projection" | "skeleton" | "" (none)


@dataclass
class ObjectGraph:
    """Node graph over tile features (no stored edges — spatial relations are
    computed geometrically from node xy/cells by the geom_compute ops at
    query time).

    ``objects_by_class`` materializes the per-class object sets
    O_n^{(c)} from the spec: keys are category strings, values are dicts
    from feature id to MapObject restricted to that category.
    """

    nodes: dict[str, MapObject]  # id -> node
    cell_size_m: float  # meters per grid cell; sourced from the tile's _meta.json
    objects_by_class: dict[str, dict[str, MapObject]] = field(default_factory=dict)
    step_index: int = 0  # n in W_n, bumped by the upsert path when threaded
    tau: float | None = None  # τ_n, wall-clock seconds of the latest update

    def objects(self, category: str) -> dict[str, MapObject]:
        """Return O_n^{(c)} for a single class. Empty dict if absent."""
        return self.objects_by_class.get(category, {})

    @property
    def classes(self) -> frozenset[str]:
        """V_n: live set of classes currently present in the graph."""
        return frozenset(self.objects_by_class.keys())

@dataclass
class WorldState:
    """W_n snapshot assembling the formal pieces in a single object."""

    grid_domain: GridDomain
    object_graph: ObjectGraph | None = None
    step_index: int = 0  # n
    tau: float | None = None  # τ_n

    @property
    def classes(self) -> frozenset[str]:
        """V_n: live class set, sourced from the feature graph."""
        if self.object_graph is None:
            return frozenset()
        return self.object_graph.classes
