#!/usr/bin/env python3
"""Bake per-block prior sidecars from OpenStreetMap.

A prior is exactly what OSM gives: the realised mask (area -> cells inside the
polygon, way -> buffered polyline, node -> its cell) clipped to the cells the
block imagery covers, ALL raw tags verbatim, the ``name`` when present, the
``osm_id``, and ``world_contour``: the source geometry's exact vertices in
CityNav local meters at 2 dp (a polygon's exterior ring closed, a way open, a
node's one vertex), which ``build_surface`` serves as the row ``mask``. Nothing
is classified, named or dropped beyond "rasterizes to no covered cell".
Every bake stamps ``meta.json`` ``bake.mode`` (and, per block, ``provenance``:
cache name + sha256, record count, the cache's georef residuals, bake time +
git sha).

The source is the shipped OSM cache per city (``tiles/osm_cache``), already
registered into CityNav local XY. The tiles dir must already hold the
``<city>_block_*_meta.json`` sidecars::

    python -m benchmarks.citynav.bake.city --city cambridge \
        --osm-cache tiles/osm_cache/cambridge_osm_local.geojson \
        --tiles-dir tiles/citynav_2m_osmv3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.citynav.eval import WorldBounds
from benchmarks.citynav.paths import TILES_DIR
from maps.sws.graph import build_object_graph, save_graph

log = logging.getLogger("bake_city")

_CITY_CHOICES = ("birmingham", "cambridge")
# The one features.json key. Every baked prior lives in this single list; the
# graph builder mints ``prior_<i>`` fids from it (provenance_index, mirroring
# ``detected_<i>``), positionally stable per the deterministic record sort.
PRIOR_KEY = "prior"
_PRIVILEGED_KEYS = {
    "ann_id",
    "ann_ids",
    "cityref_id",
    "description",
    "descriptions",
    "gt",
    "gt_position",
    "marker_positions",
    "object_id",
    "object_ids",
    "sam_label",
    "target_id",
    "target_object_id",
    "target_object_ids",
    "target_position",
    "target_positions",
}


class BakeError(RuntimeError):
    """User-facing bake failure."""


def _raster_bounds(block: str) -> tuple[float, float, float, float] | None:
    """World extent of a block's 0.1 m/px raster, or ``None`` when unreadable.

    The CityNav TIFFs carry no CRS, but they do carry an affine transform, so
    the local-XY extent is exact. That extent is the block's imagery — the
    same line the raw point cloud and the baked ortho stop at.
    """
    from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR

    tif = Path(CITYNAV_NATIVE_RGBD_DIR) / f"{block}.tif"
    if not tif.is_file():
        return None
    try:
        import rasterio  # noqa: PLC0415  (ships with the ortho extra)

        with rasterio.open(tif) as ds:
            b = ds.bounds
        return (float(b.left), float(b.bottom), float(b.right), float(b.top))
    except Exception as exc:  # noqa: BLE001 — a bad raster falls back to meta
        log.warning("%s: could not read raster bounds (%s)", block, exc)
        return None


def _load_city_meta(city: str, tiles_dir: Path, *, imagery: bool = True
                    ) -> tuple[dict[str, Any], WorldBounds, tuple[int, int], float]:
    """Meta + the geometry a block bakes over.

    Bounds come from the block's RASTER: the sidecar's box runs ~50 m past the
    imagery on every side, a ring of candidates nothing can observe (42.7% of
    all graph nodes, measured). The sidecar's box stands only when no raster
    is readable or the block has no CityNav imagery (``imagery=False``).
    """
    meta_path = tiles_dir / f"{city}_meta.json"
    if not meta_path.is_file():
        raise BakeError(
            f"missing {meta_path}; the CityNav prior bake needs an existing "
            "meta sidecar for the cell size (bounds are derived from the "
            "block raster)"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    b = meta.get("bounds") or {}
    try:
        cell_size_m = float(meta.get("cell_size_m", 2.0))
        rb = _raster_bounds(city) if imagery else None
        if rb is not None:
            x_min, y_min, x_max, y_max = rb
            # Rows run north-to-south, so grid_shape is (rows, cols) = (y, x).
            grid_shape = (
                int(round((y_max - y_min) / cell_size_m)),
                int(round((x_max - x_min) / cell_size_m)),
            )
        else:
            x_min, y_min = float(b["x_min"]), float(b["y_min"])
            x_max, y_max = float(b["x_max"]), float(b["y_max"])
            grid_shape = (int(meta["grid_shape"][0]), int(meta["grid_shape"][1]))
        bounds = WorldBounds(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise BakeError(f"{meta_path} is missing valid bounds/grid metadata") from exc
    meta["bounds"] = {
        "x_min": bounds.x_min,
        "y_min": bounds.y_min,
        "x_max": bounds.x_max,
        "y_max": bounds.y_max,
    }
    meta["grid_shape"] = [grid_shape[0], grid_shape[1]]
    meta["bounds_source"] = "block raster" if rb is not None else "meta sidecar"
    return meta, bounds, grid_shape, cell_size_m


def _cell_coverage(
    block: str,
    meta: dict[str, Any],
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
    *,
    imagery: bool = True,
) -> np.ndarray | None:
    """Boolean ``grid_shape`` mask of where the block's IMAGERY has pixels.

    Up to a third of a block's bounds rectangle is no-data a prior could sit
    over unseen. A cell is covered iff any pixel of the filled ortho
    (``load_ortho_or_rasters``) under it has ``alpha > 0``, the same no-data
    rule the runtime observation uses. Ortho rows run north-down, cell rows
    south-up, so the alpha is flipped before the reduce. ``None`` (stamped in
    ``meta["coverage_source"]``) when no ortho resolves, its geometry
    disagrees with the block bounds, or the block has no CityNav imagery
    (``imagery=False``, Cal City): the rectangle is then the rule.
    """
    meta["coverage_source"] = "bounds rectangle"
    if not imagery:
        return None
    from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR
    from maps.perception.ortho.camera_model import load_ortho_or_rasters

    try:
        ortho = load_ortho_or_rasters(block, rgbd_dir=str(CITYNAV_NATIVE_RGBD_DIR))
    except Exception as exc:  # noqa: BLE001 — a city aggregate has no raster
        log.warning(
            "%s: no ortho readable (%s); coverage stays the bounds rectangle",
            block, exc,
        )
        return None
    if ortho is None:
        log.warning("%s: no ortho found; coverage stays the bounds rectangle", block)
        return None
    ppm = float(ortho["px_per_m"])
    step = int(round(cell_size_m * ppm))
    px_tol = 0.5 / ppm
    if (
        step < 1
        or abs(cell_size_m * ppm - step) > 1e-6
        or abs(float(ortho["x_min"]) - bounds.x_min) > px_tol
        or abs(float(ortho["y_min"]) - bounds.y_min) > px_tol
    ):
        log.warning(
            "%s: ortho geometry (origin %.2f,%.2f @ %.3g px/m) does not align "
            "with the block grid (origin %.2f,%.2f @ %.3g m cells); coverage "
            "stays the bounds rectangle",
            block, float(ortho["x_min"]), float(ortho["y_min"]), ppm,
            bounds.x_min, bounds.y_min, cell_size_m,
        )
        return None
    alpha = (np.asarray(ortho["alpha"]) > 0)[::-1]  # south-up, matching cell rows
    H, W = grid_shape
    pad_r = max(0, H * step - alpha.shape[0])
    pad_c = max(0, W * step - alpha.shape[1])
    if pad_r or pad_c:  # a rounded-up grid edge holds no imagery
        alpha = np.pad(alpha, ((0, pad_r), (0, pad_c)))
    alpha = alpha[: H * step, : W * step]
    coverage = alpha.reshape(H, step, W, step).any(axis=(1, 3))
    meta["coverage_source"] = "ortho alpha"
    log.info(
        "%s: imagery covers %.1f%% of the bounds rectangle (%d/%d cells)",
        block, 100.0 * coverage.mean(), int(coverage.sum()), coverage.size,
    )
    return coverage


def _normalise_props(raw: dict[str, Any] | None) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        clean = str(key).replace(":", "_").strip()
        if not clean:
            continue
        if clean == "source":
            # OSM's own `source` tag (imagery/survey provenance, e.g. "Bing")
            # rides as `source_tag`: `source` is the bake's channel, "osm".
            clean = "source_tag"
        props[clean] = value
    return props


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _as_point_list(coords: Any) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y, *_rest in coords]


def _iter_geometry_parts(geometry: dict[str, Any]) -> Iterable[tuple[str, Any]]:
    geom_type = str(geometry.get("type") or "")
    coords = geometry.get("coordinates")
    if not geom_type or coords is None:
        return
    if geom_type in {"Point", "LineString", "Polygon"}:
        yield geom_type, coords
    elif geom_type == "MultiPoint":
        for point in coords:
            yield "Point", point
    elif geom_type == "MultiLineString":
        for line in coords:
            yield "LineString", line
    elif geom_type == "MultiPolygon":
        for poly in coords:
            yield "Polygon", poly


def _record_bbox(geometry: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """``(x_min, y_min, x_max, y_max)`` over every vertex of every part."""
    points: list[tuple[float, float]] = []
    for geom_type, coords in _iter_geometry_parts(geometry):
        chains = [[coords]] if geom_type == "Point" else [coords] if geom_type == "LineString" else coords
        for chain in chains:
            points += _as_point_list(chain)
    if not points:
        return None
    xs = [x for x, _y in points]
    ys = [y for _x, y in points]
    return min(xs), min(ys), max(xs), max(ys)


def _chain_extent(chain: Sequence[tuple[float, float]]) -> float:
    """How much of the world a chain covers: enclosed area if it closes, else
    path length, else 0 for a lone vertex. The one measure that ranks the
    parts of a multi-part geometry without asking which kind they are (they
    are all the same kind — a MultiPolygon holds polygons)."""
    if len(chain) >= 4 and chain[0] == chain[-1]:
        area = 0.0
        for i in range(len(chain) - 1):
            (x0, y0), (x1, y1) = chain[i], chain[i + 1]
            area += x0 * y1 - x1 * y0
        return abs(area) * 0.5
    return sum(math.dist(chain[i], chain[i + 1])
               for i in range(len(chain) - 1))


def _world_contour_from_geometry(geometry: dict[str, Any]) -> list[list[float]] | None:
    """The feature's EXACT vertex chain in world meters at 2 dp, or ``None``.

    A Polygon's exterior ring CLOSED (how a reader tells a ring from a way,
    ``geom_compute.is_closed_chain``), a LineString's vertices open, a Point's
    single vertex; a multi-part geometry contributes its largest part by
    ``_chain_extent``. The chain is kept whole: ``_mask_fields`` prefers it
    over the cell boundary, so it IS the footprint (``footprint_area_m2``) of
    every areal prior. A 16-vertex stride would cost a median 13.4% of the
    area on the 2.3% of Cal City rings past 16 vertices, at worst 99.1%.
    """
    best: list[tuple[float, float]] | None = None
    best_extent = -1.0
    for geom_type, coords in _iter_geometry_parts(geometry):
        if not coords:
            continue
        if geom_type == "Point":
            chain = _as_point_list([coords])
        elif geom_type == "LineString":
            chain = _as_point_list(coords)
        else:
            ring = _as_point_list(coords[0])
            if len(ring) >= 2 and ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                continue
            chain = ring + [ring[0]]
        if not chain:
            continue
        if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in chain):
            # A non-finite vertex would ride into features.json as literal
            # NaN (invalid strict JSON) and poison the shape scalars.
            continue
        extent = _chain_extent(chain)
        if extent > best_extent:
            best = chain
            best_extent = extent
    if best is None:
        return None
    closed = len(best) >= 4 and best[0] == best[-1]
    out = [[round(float(x), 2), round(float(y), 2)] for x, y in best]
    if closed:
        out = out[:-1]
    # Rounding to 2 dp can collapse neighbours onto each other; drop the
    # repeats so the closing vertex stays the chain's ONLY one. A chain that
    # rounds down to a single vertex IS that place.
    dedup = [out[0]]
    for p in out[1:]:
        if p != dedup[-1]:
            dedup.append(p)
    if not closed:
        return dedup
    # A ring is only a ring while it encloses: three distinct vertices, plus
    # the closing one that says so.
    return dedup + [list(dedup[0])] if len(dedup) >= 3 else None


def _cell_for_xy(
    x: float, y: float, bounds: WorldBounds, grid_shape: tuple[int, int], cell_size_m: float
) -> tuple[int, int] | None:
    H, W = grid_shape
    col = int((x - bounds.x_min) / cell_size_m)
    row = int((y - bounds.y_min) / cell_size_m)
    if 0 <= row < H and 0 <= col < W:
        return row, col
    return None


def _point_to_bounds_dist(x: float, y: float, bounds: WorldBounds) -> float:
    """Distance from ``(x, y)`` to the block raster rectangle, ``0.0`` inside
    it: the outside-the-rectangle arm of the ``off_raster`` flag."""
    dx = max(bounds.x_min - x, 0.0, x - bounds.x_max)
    dy = max(bounds.y_min - y, 0.0, y - bounds.y_max)
    return float(math.hypot(dx, dy))


def _window_centers(
    points: Sequence[tuple[float, float]],
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
    pad_m: float = 0.0,
) -> tuple[int, int, np.ndarray, np.ndarray] | None:
    """The cell window over ``points``' bbox (grown by ``pad_m``): its first
    row and column, and the cell-centre x per column and y per row."""
    H, W = grid_shape
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    c0 = max(0, int(math.floor((min(xs) - pad_m - bounds.x_min) / cell_size_m)))
    c1 = min(W - 1, int(math.floor((max(xs) + pad_m - bounds.x_min) / cell_size_m)))
    r0 = max(0, int(math.floor((min(ys) - pad_m - bounds.y_min) / cell_size_m)))
    r1 = min(H - 1, int(math.floor((max(ys) + pad_m - bounds.y_min) / cell_size_m)))
    if c1 < c0 or r1 < r0:
        return None
    cx = bounds.x_min + (np.arange(c0, c1 + 1) + 0.5) * cell_size_m
    cy = bounds.y_min + (np.arange(r0, r1 + 1) + 0.5) * cell_size_m
    return r0, c0, cx, cy


def _in_ring(cx: np.ndarray, cy: np.ndarray, ring: Sequence[tuple[float, float]]) -> np.ndarray:
    """Even-odd crossing test of every (row, col) cell centre against ``ring``."""
    inside = np.zeros((cy.size, cx.size), dtype=bool)
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        crosses = (yi > cy) != (yj > cy)
        x_cross = (xj - xi) * (cy - yi) / ((yj - yi) or 1e-12) + xi
        inside ^= crosses[:, None] & (cx[None, :] < x_cross[:, None])
        j = i
    return inside


def _near_polyline(
    cx: np.ndarray, cy: np.ndarray, points: Sequence[tuple[float, float]], buffer_m: float
) -> np.ndarray:
    """Cells whose centre lies within ``buffer_m`` of any segment of ``points``."""
    px, py = np.meshgrid(cx, cy)
    near = np.zeros(px.shape, dtype=bool)
    for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
        abx = bx - ax
        aby = by - ay
        denom = abx * abx + aby * aby
        if denom <= 0.0:
            dx, dy = px - ax, py - ay
        else:
            t = ((px - ax) * abx + (py - ay) * aby) / denom
            t = np.maximum(0.0, np.minimum(1.0, t))
            dx, dy = px - (ax + t * abx), py - (ay + t * aby)
        dist = np.hypot(dx, dy)
        # ponytail: np.hypot and math.hypot differ by an ulp on ~0.6% of inputs, so
        # math.hypot decides every cell this close to the buffer edge.
        for k in np.flatnonzero(np.abs(dist - buffer_m) <= 1e-9 * buffer_m):
            dist.flat[k] = math.hypot(dx.flat[k], dy.flat[k])
        near |= dist <= buffer_m
    return near


def _cells_for_geometry(
    geometry: dict[str, Any],
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
    line_buffer_m: float,
) -> set[tuple[int, int]]:
    """The uniform mask realizer: area -> cells whose centre is inside the
    polygon (holes excluded), way -> cells whose centre is within
    ``line_buffer_m`` of the polyline, node -> its cell. An area or way that
    claims no centre keeps the cells its vertices fall in."""
    cells: set[tuple[int, int]] = set()
    for geom_type, coords in _iter_geometry_parts(geometry):
        if geom_type == "Point":
            x, y, *_rest = coords
            cell = _cell_for_xy(float(x), float(y), bounds, grid_shape, cell_size_m)
            if cell is not None:
                cells.add(cell)
            continue
        if geom_type == "LineString":
            vertices = _as_point_list(coords)
            if len(vertices) < 2:
                continue
            window = _window_centers(vertices, bounds, grid_shape, cell_size_m, line_buffer_m)
            if window is not None:
                r0, c0, cx, cy = window
                mask = _near_polyline(cx, cy, vertices, line_buffer_m)
        else:
            rings = [_as_point_list(ring) for ring in coords if ring]
            if not rings or len(rings[0]) < 3:
                continue
            vertices = rings[0]
            window = _window_centers(vertices, bounds, grid_shape, cell_size_m)
            if window is not None:
                r0, c0, cx, cy = window
                mask = _in_ring(cx, cy, vertices)
                for hole in rings[1:]:
                    mask &= ~_in_ring(cx, cy, hole)
        part: set[tuple[int, int]] = set()
        if window is not None:
            rr, cc = np.nonzero(mask)
            part = set(zip((rr + r0).tolist(), (cc + c0).tolist()))
        if not part:
            for x, y in vertices:
                cell = _cell_for_xy(x, y, bounds, grid_shape, cell_size_m)
                if cell is not None:
                    part.add(cell)
        cells.update(part)
    return cells


def _geometry_kind(geometry: dict[str, Any]) -> str:
    """The mask kind the realizer produces for this geometry: ``area`` when any
    polygon part exists, else ``way`` for a line, else ``node``."""
    kinds = {g for g, _c in _iter_geometry_parts(geometry)}
    if "Polygon" in kinds:
        return "area"
    if "LineString" in kinds:
        return "way"
    return "node"


def _tag_value(value: Any) -> str | list[str] | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        return text or None
    if isinstance(value, list):
        vals = sorted({str(v).strip() for v in value if str(v).strip()})
        return vals or None
    return None


def _feature_entry(
    props: dict[str, Any],
    cells: set[tuple[int, int]],
    source_id: str | None,
    world_contour: list[list[float]] | None,
    coverage: np.ndarray | None,
) -> dict[str, Any]:
    """One baked prior: mask + ALL raw tags verbatim + name-when-present +
    osm_id. Nothing else is minted — no class, no role, no synthesized name.

    The centroid is the cell mean; when a coverage-clipped mask is concave
    the mean can land over no-data, so it snaps to the nearest kept cell —
    the centroid is what becomes the surface xy and the viewpoint target,
    and both must sit over imagery."""
    ordered_cells = [[int(r), int(c)] for r, c in sorted(cells)]
    arr = np.asarray(ordered_cells, dtype=np.float64)
    cr = float(arr[:, 0].mean())
    cc = float(arr[:, 1].mean())
    if coverage is not None and not coverage[int(round(cr)), int(round(cc))]:
        cr, cc = min(cells, key=lambda rc: (rc[0] - cr) ** 2 + (rc[1] - cc) ** 2)
    entry: dict[str, Any] = {
        "cells": ordered_cells,
        "centroid_row": round(cr, 1),
        "centroid_col": round(cc, 1),
        "source": "osm",
    }
    name = _safe_text(props.get("name"))
    if name:
        entry["name"] = name
    if world_contour:
        entry["world_contour"] = world_contour
    if source_id:
        entry["osm_id"] = source_id
    for key, value in sorted(props.items()):
        key_norm = str(key).replace(":", "_")
        if key_norm in _PRIVILEGED_KEYS:
            continue
        tag_val = _tag_value(value)
        if tag_val is not None:
            entry.setdefault(key_norm, tag_val)
    return entry


def _record_sort_key(record: tuple[dict[str, Any], dict[str, Any]]) -> tuple[Any, ...]:
    props, geometry = record
    coords = list(_iter_geometry_parts(geometry))
    first = coords[0][1] if coords else []
    text = json.dumps(first, sort_keys=True)
    return (
        _safe_text(props.get("name")) or "",
        _safe_text(props.get("osm_id")) or _safe_text(props.get("@id")) or "",
        text,
    )


def _tag_census(entries: list[dict[str, Any]]) -> dict[str, int]:
    """Top ``key=value`` tag counts over baked entries — the world-graph
    discovery census. Facts only; geometry bookkeeping keys excluded."""
    skip = {"cells", "centroid_row", "centroid_col", "world_contour",
            "off_raster", "off_raster_m", "name", "osm_id", "source"}
    counts: dict[str, int] = {}
    for entry in entries:
        for key, val in entry.items():
            if key in skip:
                continue
            vals = val if isinstance(val, list) else [val]
            for v in vals:
                pair = f"{key}={v}"
                counts[pair] = counts.get(pair, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    return dict(top)


def _features_from_records(
    records: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
    coverage: np.ndarray | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Rasterize records into ``bounds``/``grid_shape`` sidecar data and its
    ``bake_counts``.

    Every record whose geometry rasterizes to >= 1 COVERED cell is kept — the
    ONLY exclusion is no-geometry-over-imagery (whole-city records against a
    block window naturally keep the block's features). ``coverage`` is the
    imagery mask from ``_cell_coverage``; a record's cells are clipped to it,
    so a straddler keeps only its on-imagery mask and anything wholly over
    no-data never becomes a prior the agent could request a viewpoint on.
    ``coverage=None`` (no ortho) keeps every rasterized cell. A record whose
    bbox misses the window (grown by the line buffer) rasterizes to nothing
    and is skipped before the work.
    """
    line_buffer_m = max(cell_size_m * 0.5, 1.0)
    pad = max(line_buffer_m, cell_size_m)
    in_window = [
        (props, geometry, bbox)
        for props, geometry in records
        if (bbox := _record_bbox(geometry)) is not None
        and bbox[0] <= bounds.x_max + pad
        and bbox[2] >= bounds.x_min - pad
        and bbox[1] <= bounds.y_max + pad
        and bbox[3] >= bounds.y_min - pad
    ]
    entries: list[dict[str, Any]] = []
    named_kept = 0
    unnamed_kept = 0
    off_coverage_dropped = 0
    kind_counts: dict[str, int] = {}
    named_priors_total = 0
    named_off_raster: list[list[Any]] = []  # [name, off_raster_m] per off prior
    if coverage is not None:
        from scipy import ndimage  # noqa: PLC0415  (ships with the ortho extra)

        # Meters from any no-coverage cell to the nearest imagery, for the
        # off_raster honesty flag below.
        no_data_dist_m = (
            ndimage.distance_transform_edt(~coverage) * cell_size_m
        )
    for props, geometry, bbox in sorted(in_window, key=lambda r: _record_sort_key(r[:2])):
        cells = _cells_for_geometry(geometry, bounds, grid_shape, cell_size_m, line_buffer_m)
        if coverage is not None and cells:
            covered = {c for c in cells if coverage[c]}
            if not covered:
                off_coverage_dropped += 1
            cells = covered
        if not cells:
            continue
        raw_name = _safe_text(props.get("name"))
        if raw_name:
            named_kept += 1
        else:
            unnamed_kept += 1
        kind = _geometry_kind(geometry)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        source_id = _safe_text(props.get("osm_id") or props.get("@id"))
        entry = _feature_entry(props, cells, source_id,
                               _world_contour_from_geometry(geometry), coverage)
        # Raster-coverage honesty: a named prior whose TRUE geometry centroid
        # sits over no imagery (outside the raster window, or inside it on
        # no-data) is SAM-invisible. Flag those rows and tally them per block.
        if raw_name:
            named_priors_total += 1
            cx = 0.5 * (bbox[0] + bbox[2])
            cy = 0.5 * (bbox[1] + bbox[3])
            cell = _cell_for_xy(cx, cy, bounds, grid_shape, cell_size_m)
            if cell is None:
                off_m = _point_to_bounds_dist(cx, cy, bounds)
            elif coverage is not None and not coverage[cell]:
                off_m = float(no_data_dist_m[cell])
            else:
                off_m = 0.0
            if off_m > 0.0:
                entry["off_raster"] = True
                entry["off_raster_m"] = round(off_m, 1)
                named_off_raster.append([raw_name, round(off_m, 1)])
        entries.append(entry)
    off_frac = (
        round(len(named_off_raster) / named_priors_total, 3)
        if named_priors_total
        else 0.0
    )
    if named_off_raster:
        log.warning(
            "%s named prior(s) of %d sit over NO imagery "
            "(frac=%.3f, SAM-invisible): %s",
            len(named_off_raster),
            named_priors_total,
            off_frac,
            ", ".join(n for n, _d in sorted(named_off_raster, key=lambda p: -p[1])[:8]),
        )
    bake_counts = {
        "named_kept": named_kept,
        "unnamed_kept": unnamed_kept,
        # The kind census: mask kinds + top tag key=value counts.
        "kind_census": {
            "mask_kinds": dict(sorted(kind_counts.items())),
            "top_tags": _tag_census(entries),
        },
        # Records dropped for rasterizing only over no-data, and which named
        # priors keep cells but center over no imagery.
        "raster_coverage": {
            "off_coverage_dropped": off_coverage_dropped,
            "named_priors_total": named_priors_total,
            "named_priors_off_raster": len(named_off_raster),
            "named_off_raster_frac": off_frac,
            "named_off_raster": sorted(named_off_raster, key=lambda p: (-p[1], p[0])),
        },
    }
    return ({PRIOR_KEY: entries} if entries else {}), bake_counts


def _read_geojson_cache(
    path: Path,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any] | None]:
    """Read a GeoJSON cache and its ``citynav_osm_cache`` metadata block (the
    georef it was registered with and its residuals). Every feature with a
    geometry is kept: no classification, no drop-gate."""
    if not path.is_file():
        raise BakeError(f"missing OSM cache {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(raw_features, list):
        raise BakeError(f"{path} is not a GeoJSON FeatureCollection")
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw in raw_features:
        if not isinstance(raw, dict):
            continue
        geometry = raw.get("geometry")
        if not isinstance(geometry, dict):
            continue
        props = _normalise_props(raw.get("properties") if isinstance(raw.get("properties"), dict) else {})
        if raw.get("id") is not None:
            props.setdefault("osm_id", raw.get("id"))
        records.append((props, geometry))
    cache_meta = payload.get("citynav_osm_cache")
    return records, cache_meta if isinstance(cache_meta, dict) else None


# Keys a previous bake may have written into meta.json. Stripped before
# re-writing so a re-bake cannot inherit a stale provenance claim; ``bake_block``
# re-asserts its own. A key an upstream caller writes (the Cal City baker's
# ``georef`` block) is that caller's contribution and passes through.
_STALE_BAKE_FLAGS = frozenset({"bake_counts", "osm_only", "provenance"})


def _bake_provenance() -> dict[str, Any]:
    """WHO/WHEN stamp every bake writes into the block meta.

    ``BAKE_GIT_SHA`` overrides for a tree without ``.git``; otherwise
    best-effort ``git rev-parse`` from this repo, else
    ``None`` (recorded, never guessed).
    """
    sha = os.environ.get("BAKE_GIT_SHA") or None
    if sha is None:
        try:
            sha = (
                subprocess.run(
                    ["git", "-C", str(Path(__file__).resolve().parent),
                     "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, timeout=10, check=True,
                ).stdout.strip()
                or None
            )
        except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
            sha = None
    return {
        "baked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bake_git_sha": sha,
    }


def _file_sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


_EXCLUDED = (
    "CityRefer objects",
    "target IDs",
    "target positions",
    "descriptions",
    "SAM outputs",
    "satellite segmentation",
    "vehicle and bike object priors",
)


def bake_block(
    name: str,
    records: list[tuple[dict[str, Any], dict[str, Any]]],
    tiles_dir: str | Path = TILES_DIR,
    *,
    imagery: bool = True,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bake ``records`` into block ``name``'s window and write its sidecars.

    ``imagery=False`` is a block with no CityNav raster: the meta sidecar's
    bounds and the bounds rectangle's coverage. A ``provenance`` dict marks a
    city-frame cache baked per block (``bake.mode`` ``osm_blocks``) and is
    stamped with WHO/WHEN; without one the mode is ``osm_cache``.
    """
    tiles_path = Path(tiles_dir).expanduser().resolve()
    meta, bounds, grid_shape, cell_size_m = _load_city_meta(name, tiles_path, imagery=imagery)
    coverage = _cell_coverage(name, meta, bounds, grid_shape, cell_size_m, imagery=imagery)
    features, bake_counts = _features_from_records(
        records, bounds=bounds, grid_shape=grid_shape, cell_size_m=cell_size_m, coverage=coverage,
    )
    if not features:
        log.warning("%s: no OSM records rasterized onto the block imagery", name)

    tiles_path.mkdir(parents=True, exist_ok=True)
    stem = tiles_path / name
    graph = build_object_graph(features, cell_size_m=cell_size_m)
    save_graph(graph, str(stem))
    Path(f"{stem}_features.json").write_text(
        json.dumps(features, separators=(",", ":")), encoding="utf-8"
    )
    out_meta = {k: v for k, v in meta.items() if k not in _STALE_BAKE_FLAGS}
    out_meta["map_name"] = name
    out_meta["block" if "_block_" in name else "city"] = name
    out_meta["grid_shape"] = [int(grid_shape[0]), int(grid_shape[1])]
    out_meta["cell_size_m"] = float(cell_size_m)
    out_meta["n_objects"] = sum(len(v) for v in features.values())
    out_meta["source"] = "osm_only"
    out_meta["osm_only"] = True
    out_meta["bake_counts"] = bake_counts
    if provenance is not None:
        out_meta["provenance"] = {**provenance, **_bake_provenance()}
    out_meta["bake"] = {
        "mode": "osm_cache" if provenance is None else "osm_blocks",
        "excluded": list(_EXCLUDED),
    }
    Path(f"{stem}_meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
    stats = {"city": name, "features": out_meta["n_objects"], "nodes": len(graph.nodes)}
    log.info("wrote %s osm_only bake: %d features, %d nodes", name, stats["features"], stats["nodes"])
    return stats


def bake_city(city: str, osm_cache: str | Path, tiles_dir: str | Path = TILES_DIR) -> dict[str, Any]:
    """Bake ``osm_cache``, city-frame records in CityNav local XY, into every
    ``<city>_block_*`` window in ``tiles_dir`` (block bounds are windows of the
    shared city frame)."""
    tiles_path = Path(tiles_dir).expanduser().resolve()
    suffix = "_meta.json"
    blocks = sorted(p.name[: -len(suffix)] for p in tiles_path.glob(f"{city}_block_*{suffix}"))
    if not blocks:
        raise BakeError(f"no {city}_block_*{suffix} under {tiles_path}")
    cache_path = Path(osm_cache).expanduser().resolve()
    records, cache_meta = _read_geojson_cache(cache_path)
    if not records:
        raise BakeError("no OSM records to bake")
    cache_info = cache_meta or {}
    provenance = {
        "prior": "osm",
        "osm_cache": cache_path.name,
        "osm_cache_sha256": _file_sha256(cache_path),
        "fetched_at_utc": cache_info.get("fetched_at_utc"),
        "records": len(records),
        "kind_counts": cache_info.get("kind_counts"),
        "georef": cache_info.get("georef_provenance"),
    }
    total = {"blocks": 0, "features": 0, "nodes": 0}
    for block in blocks:
        stats = bake_block(block, records, tiles_path, provenance=provenance)
        total["blocks"] += 1
        total["features"] += int(stats["features"])
        total["nodes"] += int(stats["nodes"])
    log.info(
        "%s osm_blocks bake: %d blocks, %d features, %d nodes",
        city, total["blocks"], total["features"], total["nodes"],
    )
    return total


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(
        description="Bake every block of --city from its OSM cache into --tiles-dir."
    )
    ap.add_argument("--city", choices=_CITY_CHOICES, required=True)
    ap.add_argument("--osm-cache", type=Path, required=True,
                    help="the city's GeoJSON cache in CityNav local XY (tiles/osm_cache)")
    ap.add_argument("--tiles-dir", type=Path, default=TILES_DIR)
    args = ap.parse_args(argv)
    try:
        bake_city(args.city, args.osm_cache, args.tiles_dir)
    except BakeError as exc:
        ap.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
