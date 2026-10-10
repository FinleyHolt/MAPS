#!/usr/bin/env python3
"""Bake CityNav sidecars from non-target prior information.

THE production prior is OSM (PRIOR_CONTRACT.md config B): raw OpenStreetMap
pulled from the internet, tags verbatim, rasterized per block. CityNav RGBD
TIFFs are in the dataset's local metric frame and are not georeferenced, so
OSM fetches require either a cache already transformed into CityNav local XY
or an explicit georef file that defines the WGS84 query bounds and the
transform into CityNav XY. Every bake stamps ``meta.json``
``bake.mode`` + ``provenance`` (prior kind, source cache name + sha256 +
fetch time + record census, georef residuals, bake time + git sha) so the
data always says what it is and where it came from.

The ALTERNATE prior — diagnostic only, behind ``--cityrefer-landmarks`` — is
the upstream CityNav landmark set: named CityRefer landmarks with dataset
IDs, descriptions, target annotations, vehicles, street furniture, height
fields, SAM outputs, and satellite segmentation stripped before graph
construction.

OSM modes:

- ``--dump-osm-cache out.geojson --georef g.json [--city c]``: fetch OSM once
  (network) and write a GeoJSON cache in CityNav local XY, with the georef
  residual summary embedded, so the bake itself can run offline.
- ``--osm-blocks --city c (--osm-cache f | --fetch-osm --georef g)``: fetch or
  load the city-frame records ONCE, then rasterize them into EVERY
  ``<city>_block_*`` window (block bounds are windows of the shared city
  frame) and write per-block sidecars with ``source_mode="osm_blocks"``.

A prior is exactly what OSM gives us: the realised mask (area -> area mask,
way -> polyline, node -> point), ALL raw OSM tags verbatim, the OSM ``name``
when one exists, and the ``osm_id``. No classification happens at bake time —
no class table, no drop-gates beyond "the geometry rasterizes to no covered
cell in this window", no synthesized names. Covered means the cell has
imagery pixels (the block ortho's filled alpha): masks are clipped to the
imagery, so nothing the observation could never show becomes a prior the
agent can request a viewpoint on. Semantics are pulled at runtime by the
agent from the tags themselves.

Every OSM feature row carries ``world_contour``: the EXACT vertices its
source geometry was given, in CityNav local meters (the georef affine is
applied at fetch/cache time), verbatim at 2 dp. A polygon's exterior ring
CLOSED, a way's vertices open, a node's single vertex — one representation,
and nothing here asks which kind of feature it belongs to. The graph builder
stamps the row's chain onto ``MapObject.world_contour`` so ``build_surface``
serves it as the row ``mask`` + shape scalars (PRIOR_CONTRACT.md config B,
contract C2).

Canonical config-B bake, cache-first (offline; covers every block of the
city, e.g. ``cambridge_block_23``). The tiles dir must already hold the
``<city>_block_*_meta.json`` sidecars (bounds/grid/cell size), e.g. seeded
from the existing bake::

    # once, on a box with network egress:
    python -m benchmarks.citynav.bake.city \
        --dump-osm-cache tiles/osm_cache/cambridge_osm_local.geojson \
        --georef benchmarks/citynav/georef/cambridge_georef.json \
        --city cambridge

    # offline bake of every cambridge block from the cache:
    python -m benchmarks.citynav.bake.city \
        --osm-blocks --city cambridge \
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
import tempfile
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.citynav.eval import WorldBounds
from benchmarks.citynav.paths import OBJECTS_JSON, TILES_DIR
from maps.sws.graph import build_object_graph, save_graph

log = logging.getLogger("bake_city")

_CITY_CHOICES = ("birmingham", "cambridge", "stanford", "roble")
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
# Anti-cheat exclusion for the CityRefer bake modes: targets exist only
# through live perception, so movable-target annotation types never become
# priors. This is the target wall, not a semantic retrieval table.
_DISALLOWED_CITYREFER_LANDMARK_TYPES = {"Car", "Bike", "StreetFurniture"}


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


def _load_city_meta(city: str, tiles_dir: Path) -> tuple[dict[str, Any], WorldBounds, tuple[int, int], float]:
    """Meta + the geometry a block bakes over.

    Bounds come from the block's RASTER, not from the sidecar. The sidecar's
    box used to run ~50 m past the imagery on every side, and OSM covers the
    planet, so the graph filled that ring with candidates nothing could ever
    observe — 42.7% of all graph nodes, measured, none of them backed by an
    upstream CityRefer object or a CityNav target. The imagery is the block,
    so the imagery sets the bounds and the grid is sized to match. Only when
    no raster is readable does the sidecar's box stand.
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
        rb = _raster_bounds(city)
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
) -> np.ndarray | None:
    """Boolean ``grid_shape`` mask of where the block's IMAGERY has pixels.

    The bounds rectangle over-admits: the PLY footprint is tighter than the
    nominal bounds, so up to a third of a block rectangle is no-data a prior
    could sit over unseen. Coverage comes from the canonical filled ortho
    (``load_ortho_or_rasters``): after interior-hole fill, ``alpha == 0`` is
    exactly the frame-connected no-data the runtime observation treats as
    invalid, so a cell is covered iff any of its ortho pixels carry imagery.
    Ortho rows run north-down (row 0 at ``y_max``); bake cell rows run
    south-up (row 0 at ``y_min``, ``_cell_for_xy``), so the alpha is flipped
    vertically before the block reduce. Returns ``None`` — and stamps
    ``meta["coverage_source"]`` accordingly — when no ortho resolves or its
    geometry disagrees with the block bounds (fieldsite and city-aggregate
    bakes keep the rectangle rule).
    """
    from benchmarks.citynav.paths import CITYNAV_NATIVE_RGBD_DIR
    from maps.perception.ortho.camera_model import load_ortho_or_rasters

    meta["coverage_source"] = "bounds rectangle"
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
            # must not collide with the bake's `source` channel, which the
            # config-B audit requires to be exactly "osm" for OSM entries.
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
    """The feature's EXACT vertex chain in world meters, or ``None``.

    Whatever OSM gives, taken as given: a Polygon's exterior ring CLOSED (it
    ends on the vertex it began, which is how a reader tells a ring from a
    way — ``geom_compute.is_closed_chain``), a LineString's vertices open, a
    Point's single vertex. A multi-part geometry contributes its largest part
    by ``_chain_extent``. Rounded to 2 dp — the format
    ``MapObject.world_contour`` / ``build_surface`` consume.

    Nothing here asks what the feature IS. The rule this replaced returned
    the ring for a polygon and ``None`` for a line and a point, so 3,413 of
    3,414 calcity buildings carried exact vertices and 0 of 1,739 roads did;
    a road's served geometry was its 1.0 m line buffer rasterized at 2 m, its
    centerline a reconstruction off that raster, and two tuned scales existed
    downstream to recover contact and separation from the reconstruction. The
    lines were never missing — the bundle hands this function 1,702 of them
    intact — they were dropped because of what they were.

    The chain is NOT decimated. It used to be cut to 16 vertices by uniform
    stride, and ``_mask_fields`` prefers this over the cell boundary, so that
    16-gon WAS the footprint for every areal prior. Measured over the
    3,963 polygon rings in the Cal City bundle: 2.3% run past 16 vertices, and
    on those the stride cost a median 13.4% of the ring's area and at worst
    99.1% — one 33,476 m2 footprint served as 294 m2. ``footprint_area_m2`` is
    a VISIBLE column the model can score_by and filter on, so that is a wrong
    number handed to the reasoner, not just a coarse picture. A uniform stride
    is the wrong instrument regardless: it keeps every Nth vertex rather than
    the ones carrying the shape.

    The detection side still decimates (``perception.synthetic``), and should:
    a contour derived from a pixel mask is a measurement whose vertex budget is
    a rendering choice. This one is the map itself.
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


def _cell_center(
    row: int, col: int, bounds: WorldBounds, cell_size_m: float
) -> tuple[float, float]:
    return (
        bounds.x_min + (col + 0.5) * cell_size_m,
        bounds.y_min + (row + 0.5) * cell_size_m,
    )


def _point_to_bounds_dist(x: float, y: float, bounds: WorldBounds) -> float:
    """Euclidean distance from ``(x, y)`` to the block raster rectangle;
    ``0.0`` when the point is inside (or on the edge of) ``bounds``.

    The outside-the-rectangle arm of the ``off_raster`` honesty flag (the
    inside-on-no-data arm reads the coverage distance transform instead).
    Coverage-honesty metadata; the world frame / affine are untouched.
    """
    dx = max(bounds.x_min - x, 0.0, x - bounds.x_max)
    dy = max(bounds.y_min - y, 0.0, y - bounds.y_max)
    return float(math.hypot(dx, dy))


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 0.0:
        return float(math.hypot(px - ax, py - ay))
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return float(math.hypot(px - cx, py - cy))


def _point_in_ring(x: float, y: float, ring: Sequence[tuple[float, float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _point_in_polygon(
    x: float,
    y: float,
    outer: Sequence[tuple[float, float]],
    holes: Sequence[Sequence[tuple[float, float]]],
) -> bool:
    if not _point_in_ring(x, y, outer):
        return False
    return not any(_point_in_ring(x, y, hole) for hole in holes)


def _bbox_to_cell_window(
    xs: Sequence[float],
    ys: Sequence[float],
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
    pad_m: float = 0.0,
) -> tuple[range, range]:
    H, W = grid_shape
    min_x, max_x = min(xs) - pad_m, max(xs) + pad_m
    min_y, max_y = min(ys) - pad_m, max(ys) + pad_m
    c0 = max(0, int(math.floor((min_x - bounds.x_min) / cell_size_m)))
    c1 = min(W - 1, int(math.floor((max_x - bounds.x_min) / cell_size_m)))
    r0 = max(0, int(math.floor((min_y - bounds.y_min) / cell_size_m)))
    r1 = min(H - 1, int(math.floor((max_y - bounds.y_min) / cell_size_m)))
    if c1 < c0 or r1 < r0:
        return range(0), range(0)
    return range(r0, r1 + 1), range(c0, c1 + 1)


def _cells_for_point(
    coords: Any, bounds: WorldBounds, grid_shape: tuple[int, int], cell_size_m: float
) -> set[tuple[int, int]]:
    x, y, *_rest = coords
    cell = _cell_for_xy(float(x), float(y), bounds, grid_shape, cell_size_m)
    return {cell} if cell is not None else set()


def _cells_for_line(
    coords: Any,
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
    buffer_m: float,
) -> set[tuple[int, int]]:
    points = _as_point_list(coords)
    if len(points) < 2:
        return set()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    rows, cols = _bbox_to_cell_window(xs, ys, bounds, grid_shape, cell_size_m, buffer_m)
    cells: set[tuple[int, int]] = set()
    segments = list(zip(points[:-1], points[1:]))
    for row in rows:
        for col in cols:
            cx, cy = _cell_center(row, col, bounds, cell_size_m)
            if any(
                _point_segment_distance(cx, cy, ax, ay, bx, by) <= buffer_m
                for (ax, ay), (bx, by) in segments
            ):
                cells.add((row, col))
    if not cells:
        for x, y in points:
            cell = _cell_for_xy(x, y, bounds, grid_shape, cell_size_m)
            if cell is not None:
                cells.add(cell)
    return cells


def _cells_for_polygon(
    coords: Any,
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
) -> set[tuple[int, int]]:
    rings = [_as_point_list(ring) for ring in coords if ring]
    if not rings or len(rings[0]) < 3:
        return set()
    outer = rings[0]
    holes = rings[1:]
    xs = [p[0] for p in outer]
    ys = [p[1] for p in outer]
    rows, cols = _bbox_to_cell_window(xs, ys, bounds, grid_shape, cell_size_m)
    cells: set[tuple[int, int]] = set()
    for row in rows:
        for col in cols:
            cx, cy = _cell_center(row, col, bounds, cell_size_m)
            if _point_in_polygon(cx, cy, outer, holes):
                cells.add((row, col))
    if not cells:
        for x, y in outer:
            cell = _cell_for_xy(x, y, bounds, grid_shape, cell_size_m)
            if cell is not None:
                cells.add(cell)
    return cells


def _cells_for_geometry(
    geometry: dict[str, Any],
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
    line_buffer_m: float,
) -> set[tuple[int, int]]:
    """The uniform mask realizer: area -> area mask, way -> buffered polyline,
    node -> point cell. One rule per geometry kind, no per-feature policy."""
    cells: set[tuple[int, int]] = set()
    for geom_type, coords in _iter_geometry_parts(geometry):
        if geom_type == "Point":
            cells.update(_cells_for_point(coords, bounds, grid_shape, cell_size_m))
        elif geom_type == "LineString":
            cells.update(_cells_for_line(coords, bounds, grid_shape, cell_size_m, line_buffer_m))
        elif geom_type == "Polygon":
            cells.update(_cells_for_polygon(coords, bounds, grid_shape, cell_size_m))
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
    world_contour: list[list[float]] | None = None,
    coverage: np.ndarray | None = None,
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
        "source": _safe_text(props.get("source")) or "osm",
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
        # _safe_text, not raw str(): a pandas-NaN name (unnamed feature on
        # the --fetch-osm gdf path) must sort as "" exactly like the cache
        # path, where _tag_value already dropped it — otherwise the two
        # bake modes order rows differently and the positional fids diverge.
        _safe_text(props.get("name")) or "",
        _safe_text(props.get("osm_id")) or _safe_text(props.get("@id")) or "",
        text,
    )


def _tag_census(
    entries: list[dict[str, Any]], *, top_n: int = 15
) -> dict[str, int]:
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
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    return dict(top)


def _features_from_records(
    records: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
    line_buffer_m: float,
    bake_counts: dict[str, Any] | None = None,
    coverage: np.ndarray | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Rasterize records into ``bounds``/``grid_shape`` sidecar data.

    Every record whose geometry rasterizes to >= 1 COVERED cell is kept — the
    ONLY exclusion is no-geometry-over-imagery (whole-city records against a
    block window naturally keep the block's features). ``coverage`` is the
    imagery mask from ``_cell_coverage``; a record's cells are clipped to it,
    so a straddler keeps only its on-imagery mask and anything wholly over
    no-data never becomes a prior the agent could request a viewpoint on.
    ``coverage=None`` (no ortho) keeps every rasterized cell. The kind census
    (mask kinds + top tag key=value counts) lands in ``bake_counts``.
    """
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
    for props, geometry in sorted(records, key=_record_sort_key):
        cells = _cells_for_geometry(
            geometry,
            bounds,
            grid_shape,
            cell_size_m,
            line_buffer_m,
        )
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
        # True polygon geometry rides ONLY on OSM records (config B, contract
        # C2). CityRefer records (source="cityrefer_landmark"/"cityrefer_
        # unnamed") stay cells-only: their contours are loose annotation
        # hulls (median 3.41x the object dims for cars), exactly the corrupt
        # geometry the OSM-direct pivot removes.
        source = _safe_text(props.get("source")) or "osm"
        world_contour = (
            _world_contour_from_geometry(geometry) if source == "osm" else None
        )
        entry = _feature_entry(props, cells, source_id, world_contour, coverage)
        # Raster-coverage honesty: a named prior whose TRUE geometry centroid
        # sits over no imagery (outside the raster window, or inside it on
        # no-data) is SAM-invisible. Flag those rows + tally per-block
        # coverage so the T-REG gate is not silently read on the favorable
        # on-imagery subset.
        if raw_name:
            named_priors_total += 1
            bbox = _record_bbox(geometry)
            if bbox is not None:
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
            "(frac=%.3f, SAM-invisible); T-REG gate should exclude them: %s",
            len(named_off_raster),
            named_priors_total,
            off_frac,
            ", ".join(n for n, _d in sorted(named_off_raster, key=lambda p: -p[1])[:8]),
        )
    if bake_counts is not None:
        bake_counts["named_kept"] = named_kept
        bake_counts["unnamed_kept"] = unnamed_kept
        # The kind census: mask kinds + top tag key=value counts.
        bake_counts["kind_census"] = {
            "mask_kinds": dict(sorted(kind_counts.items())),
            "top_tags": _tag_census(entries),
        }
        # Per-block raster-coverage report (rides into meta.json flags):
        # records dropped for rasterizing only over no-data, and which named
        # priors keep cells but center over no imagery.
        bake_counts["raster_coverage"] = {
            "off_coverage_dropped": off_coverage_dropped,
            "named_priors_total": named_priors_total,
            "named_priors_off_raster": len(named_off_raster),
            "named_off_raster_frac": off_frac,
            "named_off_raster": sorted(named_off_raster, key=lambda p: (-p[1], p[0])),
        }
    return {PRIOR_KEY: entries} if entries else {}


def _load_cityrefer_objects(path: Path) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise BakeError(f"{path} is not a CityRefer objects.json dict")
    out: dict[str, list[dict[str, Any]]] = {}
    for map_name, entries in raw.items():
        values = entries.values() if isinstance(entries, dict) else entries
        out[str(map_name)] = [dict(obj) for obj in values if isinstance(obj, dict)]
    return out


def _cityrefer_area(obj: dict[str, Any]) -> float:
    contour = obj.get("contour") or []
    if isinstance(contour, list) and len(contour) >= 3:
        area = 0.0
        pts = [(float(p[0]), float(p[1])) for p in contour]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
            area += x0 * y1 - x1 * y0
        return abs(area) * 0.5
    dim = obj.get("dimension") or [0.0, 0.0]
    try:
        return abs(float(dim[0]) * float(dim[1]))
    except (TypeError, ValueError, IndexError):
        return 0.0


def _cityrefer_geometry(obj: dict[str, Any]) -> dict[str, Any] | None:
    contour = obj.get("contour") or []
    if isinstance(contour, list) and len(contour) >= 3:
        ring = [[float(p[0]), float(p[1])] for p in contour]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        return {"type": "Polygon", "coordinates": [ring]}
    pos = obj.get("position") or []
    dim = obj.get("dimension") or []
    try:
        cx, cy = float(pos[0]), float(pos[1])
        dx, dy = float(dim[0]), float(dim[1])
    except (TypeError, ValueError, IndexError):
        return None
    x0, x1 = cx - dx / 2.0, cx + dx / 2.0
    y0, y1 = cy - dy / 2.0, cy + dy / 2.0
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


def _cityrefer_landmarks(objects: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for obj in objects:
        name = _safe_text(obj.get("name"))
        if not name:
            continue
        object_type = str(obj.get("object_type") or obj.get("type") or "")
        if object_type in _DISALLOWED_CITYREFER_LANDMARK_TYPES:
            continue
        key = name.casefold()
        current = by_name.get(key)
        if current is None or _cityrefer_area(current) < _cityrefer_area(obj):
            by_name[key] = obj
    return sorted(
        by_name.values(),
        key=lambda obj: str(obj.get("name") or "").casefold(),
    )


def _cityrefer_unnamed(
    objects: Sequence[dict[str, Any]],
    *,
    drop_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Unnamed CityRefer entities admissible as prior anchors.

    Anti-cheat types (Car/Bike/StreetFurniture) stay excluded — targets
    exist only through live perception. Entities are returned in stable
    ``id``-sorted order so bakes of the same objects.json order identically.
    """
    kept: list[dict[str, Any]] = []
    for obj in objects:
        if _safe_text(obj.get("name")):
            continue
        object_type = str(obj.get("object_type") or obj.get("type") or "")
        if object_type in _DISALLOWED_CITYREFER_LANDMARK_TYPES:
            if drop_counts is not None:
                drop_counts[f"anticheat:{object_type}"] = (
                    drop_counts.get(f"anticheat:{object_type}", 0) + 1
                )
            continue
        if _cityrefer_geometry(obj) is None:
            if drop_counts is not None:
                drop_counts[f"no_geometry:{object_type}"] = (
                    drop_counts.get(f"no_geometry:{object_type}", 0) + 1
                )
            continue
        kept.append(obj)
    kept.sort(key=lambda obj: str(obj.get("id") or ""))
    return kept


def _features_from_cityrefer_landmarks(
    objects: Sequence[dict[str, Any]],
    *,
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
    dedupe_names: bool = True,
    include_unnamed: bool = True,
    bake_counts: dict[str, Any] | None = None,
    coverage: np.ndarray | None = None,
) -> dict[str, list[dict[str, Any]]]:
    landmark_objects = _cityrefer_landmarks(objects) if dedupe_names else list(objects)
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    n_named = 0
    for obj in landmark_objects:
        geometry = _cityrefer_geometry(obj)
        if geometry is None:
            continue
        n_named += 1
        records.append(
            (
                {
                    "name": str(obj.get("name") or "").strip(),
                    "object_type": str(obj.get("object_type") or obj.get("type") or ""),
                    "source": "cityrefer_landmark",
                },
                geometry,
            )
        )

    n_unnamed = 0
    drop_counts: dict[str, int] = {}
    if include_unnamed:
        for obj in _cityrefer_unnamed(objects, drop_counts=drop_counts):
            geometry = _cityrefer_geometry(obj)
            assert geometry is not None  # _cityrefer_unnamed already filtered
            n_unnamed += 1
            records.append(
                (
                    {
                        "object_type": str(
                            obj.get("object_type") or obj.get("type") or ""
                        ),
                        "source": "cityrefer_unnamed",
                    },
                    geometry,
                )
            )

    if bake_counts is not None:
        bake_counts["unnamed_dropped"] = drop_counts
    log.info(
        "cityrefer features: named=%d unnamed=%d dropped=%s",
        n_named,
        n_unnamed,
        drop_counts or "{}",
    )
    return _features_from_records(
        records,
        bounds=bounds,
        grid_shape=grid_shape,
        cell_size_m=cell_size_m,
        line_buffer_m=max(cell_size_m * 0.5, 1.0),
        bake_counts=bake_counts,
        coverage=coverage,
    )


def _read_geojson_cache(
    path: Path,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any] | None]:
    """Read a GeoJSON cache; also return the ``citynav_osm_cache`` metadata
    block that ``dump_osm_cache`` embeds (georef + residual provenance).
    Every feature with a geometry is kept — no classification, no drop-gate."""
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


def _read_geojson_records(path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return _read_geojson_cache(path)[0]


def _read_geopandas_records(path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise BakeError(
            "GeoPackage/cache formats other than GeoJSON require the geo extra "
            "(geopandas/osmnx/rasterio/pyproj/shapely)"
        ) from exc

    gdf = gpd.read_file(path)
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = _normalise_props({k: row[k] for k in row.index if k != "geometry"})
        props.setdefault("osm_id", str(idx))
        records.append((props, geom.__geo_interface__))
    return records


def _read_osm_cache(path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not path.is_file():
        raise BakeError(f"missing OSM cache {path}")
    if path.suffix.casefold() in {".json", ".geojson"}:
        return _read_geojson_records(path)
    return _read_geopandas_records(path)


def _validate_georef(path: Path) -> dict[str, Any]:
    try:
        georef = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BakeError(f"unable to read georef {path}: {exc}") from exc
    required = {"query_bbox_wgs84", "osm_crs", "citynav_from_osm_affine"}
    missing = sorted(required - set(georef))
    if missing:
        raise BakeError(f"{path} missing required georef field(s): {', '.join(missing)}")
    affine = georef["citynav_from_osm_affine"]
    bbox = georef["query_bbox_wgs84"]
    if not isinstance(affine, list) or len(affine) != 6:
        raise BakeError(f"{path} citynav_from_osm_affine must be a 6-number list")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise BakeError(f"{path} query_bbox_wgs84 must be [west, south, east, north]")
    return georef


def _georef_provenance(georef_path: Path) -> dict[str, Any]:
    """Residual summary from the sibling ``<stem>_report.json`` (the
    committed acceptance report of the per-city georeference fit).

    Recorded into per-block ``meta.json`` flags so every OSM bake carries the
    registration quality it was built on. Missing report is non-fatal (the
    georef itself already passed ``_validate_georef``).
    """
    report_path = georef_path.with_name(f"{georef_path.stem}_report.json")
    prov: dict[str, Any] = {"georef_file": georef_path.name}
    if not report_path.is_file():
        prov["report"] = "missing"
        return prov
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prov["report"] = "unreadable"
        return prov
    fit = report.get("fit") or {}
    prov.update(
        {
            "report_file": report_path.name,
            "passed": report.get("passed"),
            "rotation_deg": fit.get("rotation_deg"),
            "scale": fit.get("scale"),
            "acceptance_fraction_5m": fit.get("acceptance_fraction_5m"),
            "loo_residual_m": fit.get("loo_residual_m"),
        }
    )
    return prov


def _apply_affine_geometry(
    geometry: dict[str, Any], affine: Sequence[float]
) -> dict[str, Any]:
    """Apply the 6-number georef affine to every coordinate of a GeoJSON
    geometry dict: ``x' = a*x + b*y + c``, ``y' = d*x + e*y + f`` (any z
    passes through). This is the ONE explicit registration of OSM source-CRS
    coordinates into CityNav local XY — pure so the fetch-record ->
    features.json -> ``MapObject.world_contour`` chain is testable without
    osmnx/shapely (T-UNIT georef round-trip)."""
    a, b, c, d, e, f = [float(v) for v in affine]

    def _pt(pt: Sequence[float]) -> list[float]:
        x, y = float(pt[0]), float(pt[1])
        return [a * x + b * y + c, d * x + e * y + f, *pt[2:]]

    def _map(coords: Any) -> Any:
        if isinstance(coords, (list, tuple)) and coords and isinstance(
            coords[0], (int, float)
        ):
            return _pt(coords)
        return [_map(part) for part in coords]

    geom_type = str(geometry.get("type") or "")
    if geom_type == "GeometryCollection":
        return {
            "type": geom_type,
            "geometries": [
                _apply_affine_geometry(g, affine)
                for g in geometry.get("geometries") or []
            ],
        }
    return {"type": geom_type, "coordinates": _map(geometry.get("coordinates") or [])}


_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _overpass_all_elements_query(
    bbox_wgs84: Sequence[float], *, timeout_s: int = 300
) -> str:
    """The unrestricted Overpass query for a WGS84 window: EVERY element
    (``nwr`` with NO tag filter), recursed down (``(._;>;)``) so way and
    relation geometries are buildable from their member nodes. The bake
    ingests exactly what OSM gives us — there is no tag-key allowlist at the
    fetch; the only exclusion anywhere in the bake is geometric
    (no-geometry-in-window). Pure string builder so the query shape is
    testable without network or osmnx."""
    left, bottom, right, top = [float(v) for v in bbox_wgs84]
    return (
        f"[out:xml][timeout:{timeout_s}];"
        f"nwr({bottom:.6f},{left:.6f},{top:.6f},{right:.6f});"
        "(._;>;);out;"
    )


def _fetch_overpass_xml(query: str, *, attempts: int = 4) -> str:
    """POST an Overpass query with plain bounded retry/backoff (429/504 are
    routine on the shared public server). Volume is handled by the generous
    server-side timeout and the retry loop — never by narrowing the query."""
    import ssl
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    import certifi

    # Pin the trust store to certifi rather than the interpreter's default:
    # the geo env's default CA path does not carry every chain, and the bake
    # has to run in whichever env has osmnx.
    ctx = ssl.create_default_context(cafile=certifi.where())
    data = urllib.parse.urlencode({"data": query}).encode()
    headers = {
        # the public instance 406es the default urllib agent
        "User-Agent": "MAPS bake_city (citynav OSM prior bake)",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    last: Exception | None = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(10.0 * attempt)
        try:
            req = urllib.request.Request(_OVERPASS_URL, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=600, context=ctx) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
    raise BakeError(f"Overpass fetch failed after {attempts} attempts: {last}")


def _fetch_osm_records(georef: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Fetch ALL tagged OSM features with geometry in the georef window.

    One unrestricted Overpass request (``_overpass_all_elements_query``), then
    ``osmnx.features_from_xml`` builds geometries for every tagged element —
    no tag filter anywhere (``tags=None`` skips osmnx's tag filtering; the
    bbox polygon keeps the fetch's geometric window)."""
    try:
        import osmnx as ox
        from osmnx import utils_geo
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise BakeError(
            "--fetch-osm requires the geo extra "
            "(osmnx/geopandas/pyproj/shapely/rasterio)"
        ) from exc

    bbox_wgs84 = [float(v) for v in georef["query_bbox_wgs84"]]
    osm_crs = str(georef["osm_crs"])
    affine = [float(v) for v in georef["citynav_from_osm_affine"]]
    xml_text = _fetch_overpass_xml(_overpass_all_elements_query(bbox_wgs84))
    with tempfile.TemporaryDirectory(prefix="bake_city_osm_") as tmp_dir:
        xml_path = Path(tmp_dir) / "overpass.osm"
        xml_path.write_text(xml_text, encoding="utf-8")
        gdf = ox.features_from_xml(
            xml_path,
            polygon=utils_geo.bbox_to_poly(tuple(bbox_wgs84)),
        )
    if gdf.empty:
        return []
    if str(gdf.crs or "EPSG:4326") != osm_crs:
        gdf = gdf.to_crs(osm_crs)
    transformer = Transformer.from_crs(osm_crs, osm_crs, always_xy=True)
    # The identity transformer validates the CRS while leaving the affine as
    # the single explicit registration into CityNav local XY.
    del transformer

    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        geom_local = _apply_affine_geometry(geom.__geo_interface__, affine)
        props = _normalise_props({k: row[k] for k in row.index if k != "geometry"})
        props.setdefault("osm_id", str(idx))
        records.append((props, geom_local))
    return records


def dump_osm_cache(
    georef_path: str | Path,
    out_path: str | Path,
    *,
    city: str | None = None,
) -> dict[str, Any]:
    """Fetch OSM via a georef and write a GeoJSON cache in CityNav local XY.

    Lets the network fetch run on a box with egress while the bake runs
    offline from the cache. The georef residual summary is
    embedded so the offline bake still records registration provenance.
    """
    georef_file = Path(georef_path).expanduser().resolve()
    georef = _validate_georef(georef_file)
    records = _fetch_osm_records(georef)
    if not records:
        raise BakeError(f"OSM fetch for {georef_file} returned no features")
    features: list[dict[str, Any]] = []
    kind_counts: dict[str, int] = {}
    for props, geometry in records:
        kind = _geometry_kind(geometry)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        props_out: dict[str, Any] = {}
        for key, value in sorted(props.items()):
            tag_val = _tag_value(value)
            if tag_val is not None:
                props_out[key] = tag_val
        features.append(
            {
                "type": "Feature",
                "id": props_out.get("osm_id"),
                "properties": props_out,
                "geometry": geometry,
            }
        )
    payload = {
        "type": "FeatureCollection",
        "citynav_osm_cache": {
            "city": city,
            "crs_note": "coordinates are CityNav local XY (georef affine applied)",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "records": len(features),
            "georef": georef,
            "georef_provenance": _georef_provenance(georef_file),
            "kind_counts": dict(sorted(kind_counts.items())),
        },
        "features": features,
    }
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    log.info("wrote OSM cache %s: %d records %s", out, len(features), kind_counts)
    return {"path": str(out), "records": len(features), "kind_counts": kind_counts}


# Flag keys a previous bake may have written into meta.json. Stripped before
# re-writing so a mode switch (cityrefer -> osm_blocks, ...) cannot inherit a
# stale provenance claim; the current bake re-asserts its own flags.
#
# EVERY member must be one this baker re-asserts through ``flags``. A key only
# an upstream caller writes is not a stale claim to drop — it is that caller's
# contribution, and stripping it deletes the key outright. ``georef`` sat here
# and was exactly that: the Cal City baker writes the AOI's UTM zone and local
# origin into the meta, then hands it to ``bake_city_from_cache``, which
# removed it again on the way out. The block never once survived a bake, and
# tour_to_tasking --coords global had nothing to read on any Cal City tile.
_STALE_BAKE_FLAGS = frozenset(
    {
        "cityrefer_landmark_priors",
        "cityrefer_unnamed_priors",
        "bake_counts",
        "osm_only",
        "provenance",
    }
)


def _bake_provenance() -> dict[str, Any]:
    """WHO/WHEN stamp every bake writes into the block meta.

    ``BAKE_GIT_SHA`` overrides for rsynced trees without ``.git`` (cluster
    bakes); otherwise best-effort ``git rev-parse`` from this repo, else
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
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_outputs(
    city: str,
    tiles_dir: Path,
    meta: dict[str, Any],
    features: dict[str, list[dict[str, Any]]],
    grid_shape: tuple[int, int],
    cell_size_m: float,
    *,
    source_mode: str,
    source: str = "osm_only",
    flags: dict[str, Any] | None = None,
    excluded: Sequence[str] = (
        "CityRefer objects",
        "target IDs",
        "target positions",
        "descriptions",
        "SAM outputs",
        "satellite segmentation",
        "vehicle and bike object priors",
    ),
) -> dict[str, Any]:
    tiles_dir.mkdir(parents=True, exist_ok=True)
    stem = tiles_dir / city
    graph = build_object_graph(features, cell_size_m=cell_size_m)
    save_graph(graph, str(stem))
    Path(f"{stem}_features.json").write_text(
        json.dumps(features, separators=(",", ":")), encoding="utf-8"
    )

    out_meta = {k: v for k, v in meta.items() if k not in _STALE_BAKE_FLAGS}
    out_meta["map_name"] = city
    if "_block_" in city:
        out_meta["block"] = city
    else:
        out_meta["city"] = city
    out_meta["grid_shape"] = [int(grid_shape[0]), int(grid_shape[1])]
    out_meta["cell_size_m"] = float(cell_size_m)
    out_meta["n_objects"] = sum(len(v) for v in features.values())
    out_meta["source"] = source
    for key, value in (flags or {}).items():
        out_meta[key] = value
    out_meta["bake"] = {
        "mode": source_mode,
        "excluded": list(excluded),
    }
    Path(f"{stem}_meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    mask_cache = Path(f"{stem}_mask_cache.npz")
    if mask_cache.exists():
        mask_cache.unlink()

    stats = {
        "city": city,
        "features": out_meta["n_objects"],
        "nodes": len(graph.nodes),
    }
    log.info(
        "wrote %s %s bake: %d features, %d nodes",
        city,
        source,
        stats["features"],
        stats["nodes"],
    )
    return stats


def bake_city_from_cache(
    city: str,
    osm_cache: str | Path,
    *,
    tiles_dir: str | Path = TILES_DIR,
    line_buffer_m: float | None = None,
) -> dict[str, Any]:
    """Bake one CityNav city from an OSM cache in CityNav local XY."""
    tiles_path = Path(tiles_dir).expanduser().resolve()
    meta, bounds, grid_shape, cell_size_m = _load_city_meta(city, tiles_path)
    coverage = _cell_coverage(city, meta, bounds, grid_shape, cell_size_m)
    records = _read_osm_cache(Path(osm_cache).expanduser().resolve())
    bake_counts: dict[str, Any] = {}
    features = _features_from_records(
        records,
        bounds=bounds,
        grid_shape=grid_shape,
        cell_size_m=cell_size_m,
        line_buffer_m=float(line_buffer_m or max(cell_size_m * 0.5, 1.0)),
        bake_counts=bake_counts,
        coverage=coverage,
    )
    return _write_outputs(
        city,
        tiles_path,
        meta,
        features,
        grid_shape,
        cell_size_m,
        source_mode="osm_cache",
        flags={"osm_only": True, "bake_counts": bake_counts},
    )


def bake_city_from_georef_fetch(
    city: str,
    georef_path: str | Path,
    *,
    tiles_dir: str | Path = TILES_DIR,
    line_buffer_m: float | None = None,
) -> dict[str, Any]:
    """Fetch OSM using an explicit registration and bake one CityNav city."""
    tiles_path = Path(tiles_dir).expanduser().resolve()
    meta, bounds, grid_shape, cell_size_m = _load_city_meta(city, tiles_path)
    coverage = _cell_coverage(city, meta, bounds, grid_shape, cell_size_m)
    georef = _validate_georef(Path(georef_path).expanduser().resolve())
    records = _fetch_osm_records(georef)
    bake_counts: dict[str, Any] = {}
    features = _features_from_records(
        records,
        bounds=bounds,
        grid_shape=grid_shape,
        cell_size_m=cell_size_m,
        line_buffer_m=float(line_buffer_m or max(cell_size_m * 0.5, 1.0)),
        bake_counts=bake_counts,
        coverage=coverage,
    )
    return _write_outputs(
        city,
        tiles_path,
        meta,
        features,
        grid_shape,
        cell_size_m,
        source_mode="fetch_osm_georef",
        flags={"osm_only": True},
    )


def _record_bbox(geometry: dict[str, Any]) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for geom_type, coords in _iter_geometry_parts(geometry):
        if geom_type == "Point":
            x, y, *_rest = coords
            xs.append(float(x))
            ys.append(float(y))
        elif geom_type == "LineString":
            for x, y in _as_point_list(coords):
                xs.append(x)
                ys.append(y)
        elif geom_type == "Polygon":
            for ring in coords:
                for x, y in _as_point_list(ring):
                    xs.append(x)
                    ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _city_block_names(city: str, tiles_dir: Path) -> list[str]:
    suffix = "_meta.json"
    names = [p.name[: -len(suffix)] for p in tiles_dir.glob(f"{city}_block_*{suffix}")]
    if not names:
        raise BakeError(f"no {city}_block_*{suffix} under {tiles_dir}")

    def _key(name: str) -> tuple[int, str]:
        tail = name.rsplit("_", 1)[-1]
        return (int(tail) if tail.isdigit() else 1 << 30, name)

    return sorted(names, key=_key)


def bake_osm_blocks(
    city: str,
    records: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    tiles_dir: str | Path = TILES_DIR,
    line_buffer_m: float | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rasterize city-frame OSM records into every ``<city>_block_*`` window.

    Block bounds are windows of the shared city frame, so the same records
    bake every block; a record outside a block's bounds rasterizes to no
    cells there and is skipped. The per-record bbox pre-filter below is a
    pure speedup on top of that invariant (34 blocks x whole-city records).
    """
    tiles_path = Path(tiles_dir).expanduser().resolve()
    blocks = _city_block_names(city, tiles_path)
    if not records:
        raise BakeError("no OSM records to bake")
    bboxes = [_record_bbox(geometry) for _props, geometry in records]

    total = {"blocks": 0, "features": 0, "nodes": 0}
    for block in blocks:
        meta, bounds, grid_shape, cell_size_m = _load_city_meta(block, tiles_path)
        coverage = _cell_coverage(block, meta, bounds, grid_shape, cell_size_m)
        buffer_m = float(line_buffer_m or max(cell_size_m * 0.5, 1.0))
        pad = max(buffer_m, cell_size_m)
        subset = [
            rec
            for rec, bbox in zip(records, bboxes)
            if bbox is not None
            and bbox[0] <= bounds.x_max + pad
            and bbox[2] >= bounds.x_min - pad
            and bbox[1] <= bounds.y_max + pad
            and bbox[3] >= bounds.y_min - pad
        ]
        bake_counts: dict[str, Any] = {}
        features = _features_from_records(
            subset,
            bounds=bounds,
            grid_shape=grid_shape,
            cell_size_m=cell_size_m,
            line_buffer_m=buffer_m,
            bake_counts=bake_counts,
            coverage=coverage,
        )
        if not features:
            log.warning("%s: no OSM records rasterized onto the block imagery", block)
        flags: dict[str, Any] = {"osm_only": True, "bake_counts": bake_counts}
        flags["provenance"] = {**(provenance or {}), **_bake_provenance()}
        stats = _write_outputs(
            block,
            tiles_path,
            meta,
            features,
            grid_shape,
            cell_size_m,
            source_mode="osm_blocks",
            flags=flags,
        )
        total["blocks"] += 1
        total["features"] += int(stats["features"])
        total["nodes"] += int(stats["nodes"])
    log.info(
        "%s osm_blocks bake: %d blocks, %d features, %d nodes",
        city,
        total["blocks"],
        total["features"],
        total["nodes"],
    )
    return total


def bake_osm_blocks_from_cache(
    city: str,
    osm_cache: str | Path,
    *,
    tiles_dir: str | Path = TILES_DIR,
    line_buffer_m: float | None = None,
) -> dict[str, Any]:
    """Per-block OSM bake from a cache already in CityNav local XY."""
    cache_path = Path(osm_cache).expanduser().resolve()
    if not cache_path.is_file():
        raise BakeError(f"missing OSM cache {cache_path}")
    if cache_path.suffix.casefold() in {".json", ".geojson"}:
        records, cache_meta = _read_geojson_cache(cache_path)
    else:
        records, cache_meta = _read_geopandas_records(cache_path), None
    cache_info = cache_meta or {}
    return bake_osm_blocks(
        city,
        records,
        tiles_dir=tiles_dir,
        line_buffer_m=line_buffer_m,
        provenance={
            "prior": "osm",
            "osm_cache": cache_path.name,
            "osm_cache_sha256": _file_sha256(cache_path),
            "fetched_at_utc": cache_info.get("fetched_at_utc"),
            "records": len(records),
            "kind_counts": cache_info.get("kind_counts"),
            "georef": cache_info.get("georef_provenance"),
        },
    )


def bake_osm_blocks_from_georef_fetch(
    city: str,
    georef_path: str | Path,
    *,
    tiles_dir: str | Path = TILES_DIR,
    line_buffer_m: float | None = None,
) -> dict[str, Any]:
    """Per-block OSM bake with an in-process fetch (needs network egress)."""
    georef_file = Path(georef_path).expanduser().resolve()
    georef = _validate_georef(georef_file)
    records = _fetch_osm_records(georef)
    return bake_osm_blocks(
        city,
        records,
        tiles_dir=tiles_dir,
        line_buffer_m=line_buffer_m,
        provenance={
            "prior": "osm",
            "osm_cache": None,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "records": len(records),
            "georef": _georef_provenance(georef_file),
        },
    )


def bake_cityrefer_landmarks(
    *,
    maps: Sequence[str] | None = None,
    objects_json: str | Path = OBJECTS_JSON,
    tiles_dir: str | Path = TILES_DIR,
    include_city_aggregates: bool = True,
) -> dict[str, Any]:
    """Bake sidecars from the full CityRefer prior set (named + unnamed).

    Named entries follow the upstream CityNav landmark policy (non-empty
    ``name``, duplicate names collapsed to the largest contour per block,
    ``source="cityrefer_landmark"``). Unnamed structural entities are ALSO
    baked, nameless, with ``source="cityrefer_unnamed"`` and their
    ``object_type`` riding as a tag. Anti-cheat types
    (Car/Bike/StreetFurniture) stay excluded: targets exist only through live
    perception. CityRefer IDs, descriptions, target positions, object IDs,
    and height annotations are stripped as before. Kept/dropped counts land
    in ``meta.json`` under ``flags.bake_counts``.
    """
    tiles_path = Path(tiles_dir).expanduser().resolve()
    objs_by_map = _load_cityrefer_objects(Path(objects_json).expanduser().resolve())
    targets = list(maps) if maps is not None else sorted(objs_by_map)
    unknown = [name for name in targets if name not in objs_by_map]
    if unknown:
        raise BakeError(f"unknown CityRefer map(s): {', '.join(unknown)}")

    excluded = [
        "CityRefer object IDs",
        "target IDs",
        "target positions",
        "marker positions",
        "object descriptions",
        "SAM outputs",
        "satellite segmentation",
        "vehicle, bike, and street-furniture object priors",
        "height and altitude annotations",
    ]
    total = {"maps": 0, "features": 0, "nodes": 0}

    def write_one(name: str, objects: Sequence[dict[str, Any]], *, dedupe_names: bool) -> None:
        meta, bounds, grid_shape, cell_size_m = _load_city_meta(name, tiles_path)
        coverage = _cell_coverage(name, meta, bounds, grid_shape, cell_size_m)
        bake_counts: dict[str, Any] = {}
        features = _features_from_cityrefer_landmarks(
            objects,
            bounds=bounds,
            grid_shape=grid_shape,
            cell_size_m=cell_size_m,
            dedupe_names=dedupe_names,
            bake_counts=bake_counts,
            coverage=coverage,
        )
        log.info("%s: bake counts %s", name, bake_counts)
        stats = _write_outputs(
            name,
            tiles_path,
            meta,
            features,
            grid_shape,
            cell_size_m,
            source_mode="cityrefer_all",
            source="cityrefer_priors",
            flags={
                "cityrefer_landmark_priors": True,
                "cityrefer_unnamed_priors": True,
                "bake_counts": bake_counts,
                "provenance": {
                    "prior": "cityrefer",
                    "objects_json": Path(objects_json).name,
                    **_bake_provenance(),
                },
            },
            excluded=excluded,
        )
        total["maps"] += 1
        total["features"] += int(stats["features"])
        total["nodes"] += int(stats["nodes"])

    for map_name in targets:
        write_one(map_name, objs_by_map[map_name], dedupe_names=True)

    if include_city_aggregates:
        target_set = set(targets)
        cities = sorted({m.split("_block_")[0] for m in targets if "_block_" in m})
        for city in cities:
            city_blocks = sorted(m for m in objs_by_map if m.startswith(f"{city}_block_"))
            if not city_blocks or not set(city_blocks).issubset(target_set):
                continue
            city_objects: list[dict[str, Any]] = []
            for block in city_blocks:
                city_objects.extend(_cityrefer_landmarks(objs_by_map[block]))
            if city_objects:
                write_one(city, city_objects, dedupe_names=False)

    return total


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Bake CityNav sidecars from defensible priors. The production "
            "prior is OSM: --dump-osm-cache to fetch, --osm-blocks "
            "--osm-cache to bake offline. --cityrefer-landmarks is the "
            "diagnostic alternate (upstream landmark subset)."
        )
    )
    ap.add_argument("--city", choices=_CITY_CHOICES, default=None)
    ap.add_argument(
        "--cityrefer-landmarks",
        action="store_true",
        help=(
            "Bake named CityRefer landmarks only. Drops descriptions, IDs, "
            "target positions, vehicles, street furniture, and height fields."
        ),
    )
    ap.add_argument(
        "--map",
        action="append",
        default=None,
        help="Specific CityNav block(s) for --cityrefer-landmarks. Repeatable.",
    )
    ap.add_argument("--objects-json", type=Path, default=OBJECTS_JSON)
    ap.add_argument("--skip-city-aggregate", action="store_true")
    ap.add_argument("--osm-cache", type=Path, default=None)
    ap.add_argument("--georef", type=Path, default=None)
    ap.add_argument("--fetch-osm", action="store_true")
    ap.add_argument(
        "--osm-blocks",
        action="store_true",
        help=(
            "Bake per-block OSM sidecars for --city: rasterize the city-frame "
            "records into every <city>_block_* window in --tiles-dir."
        ),
    )
    ap.add_argument(
        "--dump-osm-cache",
        type=Path,
        default=None,
        metavar="OUT_GEOJSON",
        help=(
            "Fetch OSM via --georef and write a GeoJSON cache in CityNav "
            "local XY (no bake). Run where egress exists; bake offline from "
            "the cache with --osm-cache."
        ),
    )
    ap.add_argument("--tiles-dir", type=Path, default=TILES_DIR)
    ap.add_argument("--line-buffer-m", type=float, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = _parser()
    args = ap.parse_args(argv)

    if args.dump_osm_cache is not None:
        if args.cityrefer_landmarks or args.osm_blocks or args.osm_cache is not None or args.fetch_osm:
            ap.error("--dump-osm-cache combines only with --georef (and optionally --city); the fetch is implied")
        if args.georef is None:
            ap.error("--dump-osm-cache requires --georef")
        try:
            dump_osm_cache(args.georef, args.dump_osm_cache, city=args.city)
        except BakeError as exc:
            ap.error(str(exc))
        return 0

    if args.cityrefer_landmarks:
        if args.osm_cache is not None or args.fetch_osm or args.georef is not None or args.osm_blocks:
            ap.error("--cityrefer-landmarks cannot be combined with OSM cache/fetch options")
        maps = args.map
        if args.city is not None and maps is None:
            objs_by_map = _load_cityrefer_objects(args.objects_json)
            maps = sorted(m for m in objs_by_map if m.startswith(f"{args.city}_block_"))
            if not maps:
                ap.error(f"no CityRefer maps found for city {args.city!r}")
        try:
            bake_cityrefer_landmarks(
                maps=maps,
                objects_json=args.objects_json,
                tiles_dir=args.tiles_dir,
                include_city_aggregates=not args.skip_city_aggregate,
            )
        except BakeError as exc:
            ap.error(str(exc))
        return 0

    if args.city is None:
        ap.error("--city is required for OSM cache/fetch modes")
    if args.map is not None:
        ap.error("--map is only used with --cityrefer-landmarks")
    if args.skip_city_aggregate:
        ap.error("--skip-city-aggregate is only used with --cityrefer-landmarks")
    if args.objects_json != OBJECTS_JSON:
        ap.error("--objects-json is only used with --cityrefer-landmarks")
    if args.osm_cache is not None and args.fetch_osm:
        ap.error("--osm-cache and --fetch-osm are mutually exclusive")
    if args.fetch_osm and args.georef is None:
        ap.error("--fetch-osm requires --georef because CityNav TIFFs have crs=None")
    if args.georef is not None and not args.fetch_osm:
        ap.error("--georef is only used with --fetch-osm")
    if args.osm_cache is None and not args.fetch_osm:
        ap.error("provide --osm-cache, or provide --georef with --fetch-osm")

    try:
        if args.fetch_osm:
            _validate_georef(args.georef)
            bake = bake_osm_blocks_from_georef_fetch if args.osm_blocks else bake_city_from_georef_fetch
            bake(
                args.city,
                args.georef,
                tiles_dir=args.tiles_dir,
                line_buffer_m=args.line_buffer_m,
            )
        else:
            bake = bake_osm_blocks_from_cache if args.osm_blocks else bake_city_from_cache
            bake(
                args.city,
                args.osm_cache,
                tiles_dir=args.tiles_dir,
                line_buffer_m=args.line_buffer_m,
            )
    except BakeError as exc:
        ap.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
