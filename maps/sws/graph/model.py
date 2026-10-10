"""Feature graph data structures.

  MapObject   — a spatial feature with geometry and metadata
  ObjectGraph — dict-backed node graph
  GridDomain  — the tile raster grid and its cell size
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GridDomain:
    """The tile raster grid: ``height`` rows, ``width`` columns, 0-indexed,
    ``cell_size_m`` meters per cell."""

    height: int
    width: int
    cell_size_m: float


@dataclass
class MapObject:
    """A feature in the spatial graph: a prebake (map) node or, with
    ``provenance="detected"``, a runtime detection with per-observation
    counters. Grid fields are (row, col) on the tile grid."""

    id: str  # "prior_5", "detected_0"
    class_name: str  # the features.json list key ("prior") or a detection's SAM label ("white car")
    name: str | None
    tags: dict[str, str]
    centroid: tuple[int, int]  # (row, col)
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    cell_count: int
    orientation: float | None  # Degrees [0, 180), 0=N-S, 90=E-W
    orientation_confidence: float = 0.0  # 0=round/ambiguous, 1=clearly linear
    provenance: str = "prebaked"  # "prebaked" | "detected"
    first_seen_t: float | None = None  # wall-clock of first observation
    last_seen_t: float | None = None
    observation_count: int = 0
    # The in-view mask pixel count of the frame the perceived color was read
    # from; a later frame replaces the color only with a larger view
    # (runtime.apply_best_view_color). 0 = no color view yet.
    best_view_px: int = 0
    confidence: float = 1.0  # aggregated detection confidence [0,1]
    # Observed attributes, {key: {value: count}}; keys are open (color, ...).
    # Prebake nodes receive only pixel-derived attributes
    # (runtime.upsert_prior_percept), never catalog text.
    semantic_attributes: dict[str, dict[str, int]] = field(default_factory=dict)
    # The world tracker's stable identity; None for prebake nodes and
    # untracked detections.
    track_id: str | None = None
    # Footprint cells, ``(N, 2) int32`` (row, col): the builder's for a prebake
    # node, the latest projected footprint for a tracked detection, None for a
    # centroid-only detection. No dense per-node raster is kept.
    cells: np.ndarray | None = None
    # The exact vertex chain in world meters, ``[[x, y], ...]``: a point, an
    # open way, or a closed ring that repeats its first vertex
    # (``geom_compute.is_closed_chain``). A measured contour is decimated by
    # its producer; a map contour is kept whole, since it is the footprint and
    # footprint_area_m2 is computed from it. build_surface prefers it over
    # ``cells`` for the row mask and shape scalars.
    world_contour: list[list[float]] | None = None
    # The tracker's sub-cell world position; geometry readers prefer it over
    # the cell centroid.
    world_xy: tuple[float, float] | None = None
    # Mask-derived geometry in the grid (row, col) frame, never world xy: the
    # builder has no tile origin, so frame-dependent quantities (bearing,
    # left/right) are derived by the tools at query time.
    # ``centerline`` is the principal-axis polyline of the mask, sampled at
    # grid resolution: a short stub for a compact mask, the centerline of an
    # elongated one.
    centerline: list[list[float]] | None = None  # [[row, col], ...] grid frame
    centerline_arc_m: list[float] | None = None  # cumulative meters per vertex
    axis_length_m: float = 0.0  # total centerline arc length, meters
    # Max perpendicular distance (meters) of a mask cell to the straight PCA
    # line: about half the width for a straight mask, more for a curved one.
    axis_residual_m: float = 0.0
    centerline_method: str = ""  # "pca_projection" | "skeleton" | "" (none)


@dataclass
class ObjectGraph:
    """Node graph over tile features, without edges: spatial relations are
    computed from node xy/cells by the geom_compute ops at query time.
    """

    nodes: dict[str, MapObject]  # id -> node
    cell_size_m: float  # meters per grid cell; sourced from the tile's _meta.json
    step_index: int = 0  # set by the upserts that are given one
    tau: float | None = None  # wall-clock seconds of the latest update
