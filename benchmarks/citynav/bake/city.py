#!/usr/bin/env python3
"""Bake per-block prior sidecars from OpenStreetMap.

A prior is exactly what OSM gives: the realised mask (area -> cells inside the
polygon, way -> buffered polyline, node -> its cell) clipped to the cells the
block imagery covers, every raw tag verbatim except the ground-truth label keys
(``_PRIVILEGED_KEYS``), the ``name`` when present, the ``osm_id``, and
``world_contour``, the source geometry's exact vertices, which
``build_surface_from_graph`` serves as the row ``mask``. Nothing is classified
or named, and no feature is dropped beyond "rasterizes to no covered cell".

The source is the shipped OSM cache per city (``tiles/osm_cache``), already
registered into CityNav local XY. Every ``<city>_block_<n>.tif`` raster
under ``SEARCH_CITYNAV_RGBD_DIR`` is a block; the tiles dir may start empty::

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
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.citynav.eval import WorldBounds
from benchmarks.citynav.paths import BAKE_OUTPUT_SUFFIXES, TILES_DIR
from maps.search.analyze_scene.geom_compute import is_closed_chain

log = logging.getLogger("bake_city")

_CITY_CHOICES = ("birmingham", "cambridge")
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


#: The cell of every CityNav block bake, metres.
CELL_SIZE_M = 2.0


def _raster_tif(block: str) -> Path:
    from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR

    return Path(CITYNAV_NATIVE_RGBD_DIR) / f"{block}.tif"


def _raster_bounds(block: str) -> tuple[float, float, float, float]:
    """World extent of a block's 0.1 m/px raster.

    The CityNav TIFFs carry no CRS, but they do carry an affine transform, so
    the local-XY extent is exact. That extent is the block's imagery: the
    same line the raw point cloud and the baked ortho stop at.
    """
    tif = _raster_tif(block)
    try:
        import rasterio  # noqa: PLC0415  (ships with the ortho extra)

        with rasterio.open(tif) as ds:
            b = ds.bounds
    except Exception as exc:  # noqa: BLE001 - every failure is the same refusal
        raise BakeError(f"{block}: cannot read the block raster {tif} ({exc})") from exc
    return (float(b.left), float(b.bottom), float(b.right), float(b.top))


def _cell_coverage(
    block: str,
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
) -> np.ndarray | None:
    """Boolean ``grid_shape`` mask of where the block's IMAGERY has pixels: a
    cell is covered iff any pixel of the filled ortho under it has
    ``alpha > 0``, the runtime observation's no-data rule. ``None`` (the
    bounds rectangle is the coverage) when the ortho's geometry disagrees
    with the block grid; an unreadable ortho is a ``BakeError``.
    """
    from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR
    from maps.perception.ortho.camera_model import load_ortho_or_rasters

    try:
        ortho = load_ortho_or_rasters(block, rgbd_dir=str(CITYNAV_NATIVE_RGBD_DIR))
    except Exception as exc:  # noqa: BLE001 - every failure is the same refusal
        raise BakeError(f"{block}: cannot read the block imagery under {CITYNAV_NATIVE_RGBD_DIR} ({exc})") from exc
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
    log.info(
        "%s: imagery covers %.1f%% of the bounds rectangle (%d/%d cells)",
        block, 100.0 * coverage.mean(), int(coverage.sum()), coverage.size,
    )
    return coverage


def normalise_props(raw: dict[str, Any] | None) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        clean = str(key).replace(":", "_").strip()
        if not clean:
            continue
        if clean == "source":
            # OSM's own `source` tag (imagery/survey provenance, e.g. "Bing")
            # rides as `source_tag`, a key the search text drops
            # (surface._UNSERVED_TAGS).
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
    path length, else 0 for a lone vertex; it ranks a multi-part geometry's
    parts."""
    if is_closed_chain(chain):
        area = 0.0
        for i in range(len(chain) - 1):
            (x0, y0), (x1, y1) = chain[i], chain[i + 1]
            area += x0 * y1 - x1 * y0
        return abs(area) * 0.5
    return sum(math.dist(chain[i], chain[i + 1])
               for i in range(len(chain) - 1))


def _world_contour_from_geometry(geometry: dict[str, Any]) -> list[list[float]] | None:
    """The feature's EXACT vertex chain in world meters at 2 dp, or ``None``.

    A Polygon's exterior ring CLOSED (how ``is_closed_chain`` tells a ring
    from a way), a LineString's vertices open, a Point's
    single vertex; a multi-part geometry contributes its largest part by
    ``_chain_extent``. The chain is kept whole: it is the footprint
    (``footprint_area_m2``) of every areal prior.
    """
    chains = []
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
        # A non-finite vertex would ride into features.json as literal NaN
        # (invalid strict JSON) and poison the shape scalars.
        if all(math.isfinite(v) for p in chain for v in p):
            chains.append(chain)
    if not chains:
        return None
    best = max(chains, key=_chain_extent)
    closed = is_closed_chain(best)
    out = [[round(float(x), 2), round(float(y), 2)] for x, y in (best[:-1] if closed else best)]
    # Rounding to 2 dp can collapse neighbours onto each other; drop
    # consecutive repeats. A chain that rounds down to a single vertex IS
    # that place.
    dedup = [p for i, p in enumerate(out) if i == 0 or p != out[i - 1]]
    if not closed:
        return dedup
    # A ring is only a ring while it encloses: three distinct vertices, plus
    # the closing one that says so.
    return dedup + [list(dedup[0])] if len(dedup) >= 3 else None


def _cell_for_xy(
    x: float, y: float, bounds: WorldBounds, grid_shape: tuple[int, int], cell_size_m: float
) -> tuple[int, int] | None:
    H, W = grid_shape
    col = math.floor((x - bounds.x_min) / cell_size_m)
    row = math.floor((y - bounds.y_min) / cell_size_m)
    if 0 <= row < H and 0 <= col < W:
        return row, col
    return None


def _point_to_bounds_dist(x: float, y: float, bounds: WorldBounds) -> float:
    """Distance from ``(x, y)`` to the block raster rectangle, ``0.0`` inside
    it: the outside-the-rectangle arm of the off-imagery warning."""
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
    def cell(v, lo):
        return int(math.floor((v - lo) / cell_size_m))

    c0, c1 = max(0, cell(min(xs) - pad_m, bounds.x_min)), min(W - 1, cell(max(xs) + pad_m, bounds.x_min))
    r0, r1 = max(0, cell(min(ys) - pad_m, bounds.y_min)), min(H - 1, cell(max(ys) + pad_m, bounds.y_min))
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


def _tag_value(value: Any) -> str | list[str] | None:
    if isinstance(value, list):
        return sorted({str(v).strip() for v in value if str(v).strip()}) or None
    return _safe_text(value) if isinstance(value, (str, int, float, bool)) else None


def _feature_entry(
    props: dict[str, Any],
    cells: set[tuple[int, int]],
    source_id: str | None,
    world_contour: list[list[float]] | None,
    coverage: np.ndarray | None,
) -> dict[str, Any]:
    """One baked prior: mask, every raw tag verbatim except
    ``_PRIVILEGED_KEYS``, the name when present and the osm_id. The centroid
    is the cell mean, snapped to the nearest kept cell when it lands over
    no-data."""
    ordered_cells = [[int(r), int(c)] for r, c in sorted(cells)]
    arr = np.asarray(ordered_cells, dtype=np.float64)
    cr = float(arr[:, 0].mean())
    cc = float(arr[:, 1].mean())
    if coverage is not None and not coverage[int(round(cr)), int(round(cc))]:
        # ponytail: an exact distance tie goes to the set's iteration order;
        # a (distance, cell) key orders it, at the cost of a new bake hash.
        cr, cc = min(cells, key=lambda rc: (rc[0] - cr) ** 2 + (rc[1] - cc) ** 2)
    entry: dict[str, Any] = {
        "cells": ordered_cells,
        "centroid_row": round(cr, 1),
        "centroid_col": round(cc, 1),
    }
    name = _safe_text(props.get("name"))
    if name:
        entry["name"] = name
    if world_contour:
        entry["world_contour"] = world_contour
    if source_id:
        entry["osm_id"] = source_id
    for key, value in sorted(props.items()):
        if key in _PRIVILEGED_KEYS:
            continue
        tag_val = _tag_value(value)
        if tag_val is not None:
            entry.setdefault(key, tag_val)
    return entry


def _record_sort_key(record: tuple[dict[str, Any], dict[str, Any]]) -> tuple[Any, ...]:
    props, geometry = record
    coords = list(_iter_geometry_parts(geometry))
    first = coords[0][1] if coords else []
    text = json.dumps(first, sort_keys=True)
    return (
        _safe_text(props.get("name")) or "",
        _safe_text(props.get("osm_id")) or "",
        text,
    )


def _features_from_records(
    records: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
    coverage: np.ndarray | None,
) -> dict[str, list[dict[str, Any]]]:
    """Rasterize records into ``bounds``/``grid_shape`` sidecar data.

    A record's cells are clipped to ``coverage`` (``None`` keeps every
    rasterized cell); every record keeping >= 1 cell is kept, nothing else is
    dropped.
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
    named_priors_total = 0
    named_off_raster: list[list[Any]] = []  # [name, metres off imagery] per off prior
    if coverage is not None:
        from scipy import ndimage  # noqa: PLC0415  (ships with the ortho extra)

        # Meters from any no-coverage cell to the nearest imagery, for the
        # off-imagery warning below.
        no_data_dist_m = (
            ndimage.distance_transform_edt(~coverage) * cell_size_m
        )
    for props, geometry, bbox in sorted(in_window, key=lambda r: _record_sort_key(r[:2])):
        cells = _cells_for_geometry(geometry, bounds, grid_shape, cell_size_m, line_buffer_m)
        if coverage is not None:
            cells = {c for c in cells if coverage[c]}
        if not cells:
            continue
        raw_name = _safe_text(props.get("name"))
        entry = _feature_entry(props, cells, _safe_text(props.get("osm_id")),
                               _world_contour_from_geometry(geometry), coverage)
        # A named prior whose TRUE geometry centroid sits over no imagery
        # (outside the raster window, or inside it on no-data) is
        # SAM-invisible; the bake warns with them.
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
                named_off_raster.append([raw_name, round(off_m, 1)])
        entries.append(entry)
    if named_off_raster:
        log.warning(
            "%s named prior(s) of %d sit over NO imagery "
            "(frac=%.3f, SAM-invisible): %s",
            len(named_off_raster),
            named_priors_total,
            round(len(named_off_raster) / named_priors_total, 3),
            ", ".join(n for n, _d in sorted(named_off_raster, key=lambda p: -p[1])[:8]),
        )
    # The graph builder mints ``prior_<i>`` fids from this key in record order.
    return {"prior": entries} if entries else {}


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
        props = normalise_props(raw.get("properties") if isinstance(raw.get("properties"), dict) else {})
        if raw.get("id") is not None:
            props.setdefault("osm_id", raw.get("id"))
        records.append((props, geometry))
    cache_meta = payload.get("citynav_osm_cache")
    return records, cache_meta if isinstance(cache_meta, dict) else None


def _file_sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def bake_block(
    name: str,
    records: list[tuple[dict[str, Any], dict[str, Any]]],
    tiles_dir: str | Path = TILES_DIR,
    *,
    bounds: WorldBounds | None = None,
    meta: dict[str, Any] | None = None,
) -> int:
    """Bake ``records`` into block ``name``'s window, write its sidecars and
    return the number of features written.

    The grid is ``CELL_SIZE_M`` cells over the block's raster extent, covered
    where its imagery has pixels; a block with no CityNav raster (Cal City)
    passes its ``bounds``, all covered. ``meta`` keys (a caller's
    ``coordinate_frame`` and ``georef``, a city-cache bake's ``provenance``)
    are written to the meta sidecar after the grid, as given.
    """
    tiles_path = Path(tiles_dir).expanduser().resolve()
    imagery = bounds is None
    if bounds is None:
        x_min, y_min, x_max, y_max = _raster_bounds(name)
        bounds = WorldBounds(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    # Row 0 is y_min (south) and rows run north, so grid_shape is (rows, cols) = (y, x).
    grid_shape = (int(round((bounds.y_max - bounds.y_min) / CELL_SIZE_M)),
                  int(round((bounds.x_max - bounds.x_min) / CELL_SIZE_M)))
    coverage = _cell_coverage(name, bounds, grid_shape, CELL_SIZE_M) if imagery else None
    features = _features_from_records(
        records, bounds=bounds, grid_shape=grid_shape, cell_size_m=CELL_SIZE_M, coverage=coverage,
    )
    if not features:
        log.warning("%s: no OSM records rasterized onto the block imagery", name)

    tiles_path.mkdir(parents=True, exist_ok=True)
    meta_path, feat_path = (Path(f"{tiles_path / name}{s}") for s in BAKE_OUTPUT_SUFFIXES)
    feat_path.write_text(
        json.dumps(features, separators=(",", ":")), encoding="utf-8"
    )
    out_meta: dict[str, Any] = {
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "cell_size_m": CELL_SIZE_M,
        "bounds": {"x_min": bounds.x_min, "y_min": bounds.y_min,
                   "x_max": bounds.x_max, "y_max": bounds.y_max},
        **(meta or {}),
    }
    meta_path.write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
    count = sum(len(v) for v in features.values())
    log.info("wrote %s bake: %d features", name, count)
    return count


def bake_city(city: str, osm_cache: str | Path, tiles_dir: str | Path = TILES_DIR) -> None:
    """Bake ``osm_cache``, city-frame records in CityNav local XY, into
    ``tiles_dir`` for every ``<city>_block_<n>.tif`` raster (block bounds are
    windows of the shared city frame)."""
    tiles_path = Path(tiles_dir).expanduser().resolve()
    rasters = _raster_tif(f"{city}_block_0").parent
    blocks = sorted(p.stem for p in rasters.glob(f"{city}_block_*.tif")
                    if re.fullmatch(rf"{city}_block_\d+", p.stem))
    if not blocks:
        raise BakeError(f"no {city}_block_<n>.tif under {rasters}")
    cache_path = Path(osm_cache).expanduser().resolve()
    records, cache_meta = _read_geojson_cache(cache_path)
    if not records:
        raise BakeError("no OSM records to bake")
    provenance = {
        "prior": "osm",
        "osm_cache": cache_path.name,
        "osm_cache_sha256": _file_sha256(cache_path),
        "records": len(records),
        "georef": (cache_meta or {}).get("georef_provenance"),
    }
    for block in blocks:
        bake_block(block, records, tiles_path, meta={"provenance": provenance})


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
