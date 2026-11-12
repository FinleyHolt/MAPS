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
    """A feature in the spatial graph, keyed by its fid ("prior_5",
    "detected_0"): a prebake (map) node or, with ``provenance="detected"``, a
    runtime detection. Grid fields are (row, col) on the tile grid."""

    class_name: str  # the features.json list key ("prior") or a detection's SAM label ("white car")
    name: str | None
    tags: dict[str, str]
    centroid: tuple[int, int]  # (row, col)
    provenance: str = "prebaked"  # "prebaked" | "detected"
    # The in-view mask pixel count of the frame the perceived color was read
    # from; a later frame replaces the color only with a larger view
    # (runtime.apply_best_view_color). 0 = no color view yet.
    best_view_px: int = 0
    # Observed attributes, {key: {value: count}}; keys are open (color, ...).
    # Prebake nodes receive only pixel-derived attributes
    # (runtime.upsert_prior_percept), never catalog text.
    semantic_attributes: dict[str, dict[str, int]] = field(default_factory=dict)
    # The world tracker's stable identity; None for prebake nodes and
    # untracked detections.
    track_id: str | None = None
    # Footprint cells, ``(N, 2) int32`` (row, col): the bake's, a tracked
    # detection's latest footprint, or None for a centroid-only detection.
    cells: np.ndarray | None = None
    # The exact vertex chain in world meters, ``[[x, y], ...]``: a point, an
    # open way, or a closed ring repeating its first vertex. A map contour is
    # kept whole (it is the footprint); the surface prefers it over ``cells``.
    world_contour: list[list[float]] | None = None
    # The tracker's sub-cell world position; geometry readers prefer it over
    # the cell centroid.
    world_xy: tuple[float, float] | None = None
    # Mask-derived geometry in the grid (row, col) frame (the builder has no
    # tile origin): the mask's principal-axis polyline at grid resolution.
    centerline: list[list[float]] | None = None  # [[row, col], ...] grid frame
    axis_length_m: float = 0.0  # total centerline arc length, meters


@dataclass
class ObjectGraph:
    """Node graph over tile features, without edges."""

    nodes: dict[str, MapObject]  # id -> node
    cell_size_m: float  # meters per grid cell; sourced from the tile's _meta.json
