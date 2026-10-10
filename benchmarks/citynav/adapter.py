"""CityNav dataset adapter.

Parses CityNav episodes (natural-language description, map id, ground-truth
target) and CityRefer landmark objects, and derives the landmark features +
ObjectGraph the runtime loads.

The CityNav data schema mirrors ``citynav-arxiv/gsamllavanav/dataset/...``:
  * per-split JSON (e.g. ``data/citynav/citynav_val_seen.json``) is a list of
    MTurk trajectory dicts with fields ``area``, ``block``, ``descriptions``,
    ``target_positions``, ``marker_positions``, ``object_ids``, ``ann_ids``,
    ``trajectory``. A row with no ``block`` names a whole-AOI map
    (``map_name_for_row``).
  * ``data/cityrefer/objects.json`` is a map ``map_name → list[CityReferObject]``,
    each object having ``id``, ``name``, ``object_type``, ``position`` (xyz center),
    ``dimension`` (xyz extent), ``contour`` (2D polygon).

Every landmark in CityRefer maps directly to an open-vocabulary label string
(snake_case: ``building``, ``road``, ``water``, ``bridge``, ``wall``, ...).
The flight runtime carries no scene-mask payload — the reasoner queries the
feature graph instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from benchmarks.citynav.eval import WorldBounds


# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------


@dataclass
class CityRefObject:
    """A single landmark / referred object from ``objects.json``."""

    map_name: str
    id: int
    name: str
    object_type: str
    position: tuple[float, float, float]  # (x, y, z) center, meters
    dimension: tuple[float, float, float]  # (dx, dy, dz) extent, meters
    contour: list[tuple[float, float]] = field(default_factory=list)  # 2D polygon
    descriptions: list[str] = field(default_factory=list)  # NL descriptors


@dataclass
class CityNavEpisode:
    """One row of a CityNav split JSON, enriched with resolved target / start."""

    map_name: str  # e.g. "birmingham_block_5"
    episode_id: str  # stable string id for logging
    description: str  # the NL instruction
    start_xy: tuple[float, float]  # first trajectory pose xy
    #: Upstream's start altitude, the row's ``trajectory[0]`` z (absolute, as
    #: every altitude here is); None on a mission flown under an operator's
    #: own words.
    start_z: float | None
    #: Mean of target_positions (xy); None on a mission flown under an
    #: operator's own words, which has no row and which the bench does not score.
    target_xy: tuple[float, float] | None
    target_object_ids: list[int]
    target_mask: list[tuple[float, float]] | None  # hand-labeled GT polygon
    marker_positions: list[tuple[float, float, float]]
    raw: dict  # underlying JSON row for provenance
    success_radius_m: float | None = None
    """This episode's fallback success radius in the dataset it was read
    from, when the split states one per row (Cal City publishes the same
    episodes as a 20 m dataset and a precise one). ``None`` leaves the run
    config's ``success_radius_m`` in charge, which is every CityNav split."""


# ---------------------------------------------------------------------------
# Open-vocab label conventions
# ---------------------------------------------------------------------------
#
# CityRefer ships 13 object_types. We emit one canonical open-vocab label per
# type. No collapsing.

OBJTYPE_TO_LABEL: dict[str, str] = {
    "Building": "building",
    "Wall": "wall",
    "Bridge": "bridge",
    "TrafficRoad": "road",
    "Rail": "rail",
    "Footpath": "footpath",
    "Parking": "parking",
    "Water": "water",
    "HighVegetation": "vegetation",
    "Ground": "ground",
    "StreetFurniture": "street_furniture",
    "Car": "car",
    "Bike": "bike",
}

def object_label(obj: CityRefObject) -> str | None:
    """Map a CityRefer object to its canonical scene label (or None)."""
    return OBJTYPE_TO_LABEL.get(obj.object_type)


# ---------------------------------------------------------------------------
# Split + objects loading
# ---------------------------------------------------------------------------


def load_split(split_file: Path) -> list[dict]:
    """Load a CityNav split JSON (list of MTurk trajectory rows)."""
    with open(split_file) as f:
        return json.load(f)


def load_objects(objects_json: Path) -> dict[str, list[CityRefObject]]:
    """Load ``cityrefer/objects.json`` into a dict keyed by SensatUrban ``map_name``.

    CityRefer stores objects as a nested dict ``{map_name: {object_id: {...}}}``
    or a flat list — we handle both defensively.
    """
    with open(objects_json) as f:
        raw = json.load(f)

    by_map: dict[str, list[CityRefObject]] = {}

    def _mk(map_name: str, obj: dict) -> CityRefObject | None:
        try:
            pos = obj["position"]
            dim = obj["dimension"]
            contour_raw = obj.get("contour", []) or []
            descriptions_raw = obj.get("descriptions") or []
            return CityRefObject(
                map_name=map_name,
                id=int(obj["id"]),
                name=str(obj.get("name") or obj.get("object_name") or ""),
                object_type=str(obj.get("object_type") or obj.get("type") or ""),
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                dimension=(float(dim[0]), float(dim[1]), float(dim[2])),
                contour=[(float(p[0]), float(p[1])) for p in contour_raw],
                descriptions=[str(d) for d in descriptions_raw if d],
            )
        except (KeyError, TypeError, ValueError):
            return None

    if isinstance(raw, dict):
        for map_name, entries in raw.items():
            objs: list[CityRefObject] = []
            iterator: Iterable[dict]
            if isinstance(entries, dict):
                iterator = entries.values()
            else:
                iterator = entries
            for obj in iterator:
                parsed = _mk(map_name, obj)
                if parsed is not None:
                    objs.append(parsed)
            by_map[map_name] = objs
    elif isinstance(raw, list):
        for obj in raw:
            map_name = str(obj.get("map_name") or obj.get("scene_id") or "")
            if not map_name:
                continue
            parsed = _mk(map_name, obj)
            if parsed is not None:
                by_map.setdefault(map_name, []).append(parsed)

    return by_map


def map_name_for_row(row: dict) -> str:
    """The ONE map-name rule for a dataset row.

    CityNav rows carry ``area`` + ``block`` and compose
    ``<area>_block_<block>``; an ``area`` that already contains "block"
    is a full tile id and wins verbatim; a row with no (or empty)
    ``block`` names a whole-AOI map — ``area`` verbatim.
    """
    area = str(row["area"])
    block = row.get("block")
    block_s = "" if block is None else str(block)
    return f"{area}_block_{block_s}" if (block_s and "block" not in area) else area


def parse_episode(row: dict, row_index: int | None = None) -> CityNavEpisode:
    """Coerce one split-JSON row into a CityNavEpisode.

    When ``row_index`` is supplied, append ``#<row_index>`` to the
    episode_id. CityNav reuses the same ``ann_ids[0]`` across many rows
    of the same block (each row carries one description on a shared
    trajectory), so the legacy ``map::ann`` form collides for ~99% of
    the corpus. The row-index suffix disambiguates those rows so every
    scenario is uniquely addressable. Callers that don't care about
    per-row identity can still omit it and get the legacy id.
    """
    map_name = map_name_for_row(row)
    descriptions = row.get("descriptions") or []
    ann_ids = row.get("ann_ids") or [0]
    base_id = f"{map_name}::{ann_ids[0]}"
    episode_id = base_id if row_index is None else f"{base_id}#{row_index}"

    target_positions = row.get("target_positions") or []
    if not target_positions:
        raise ValueError(f"Episode {episode_id} has no target_positions")
    tx = float(np.mean([p[0] for p in target_positions]))
    ty = float(np.mean([p[1] for p in target_positions]))

    trajectory = row.get("trajectory") or []
    if not trajectory:
        raise ValueError(f"Episode {episode_id} has no trajectory")
    start_xy = (float(trajectory[0][0]), float(trajectory[0][1]))

    markers = [
        (float(p[0]), float(p[1]), float(p[2]))
        for p in (row.get("marker_positions") or [])
    ]
    _raw_mask = row.get("target_mask")
    target_mask = (
        [(float(p[0]), float(p[1])) for p in _raw_mask]
        if _raw_mask
        else None
    )

    _raw_radius = row.get("success_radius_m")

    return CityNavEpisode(
        map_name=map_name,
        episode_id=episode_id,
        description=descriptions[0] if descriptions else "",
        start_xy=start_xy,
        start_z=float(trajectory[0][2]),
        target_xy=(tx, ty),
        target_object_ids=list(row.get("object_ids") or []),
        target_mask=target_mask,
        marker_positions=markers,
        raw=row,
        success_radius_m=(
            float(_raw_radius) if _raw_radius is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Features dict + ObjectGraph from CityRefer landmarks
# ---------------------------------------------------------------------------


def rasterize_polygon(
    polygon: list[tuple[float, float]] | list[list[float]],
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
) -> list[list[int]]:
    """Rasterize a world-coordinate polygon into ``[[row, col], ...]`` cells,
    via cell-center containment (matplotlib Path). Empty on a degenerate
    polygon (<3 points) or a missing matplotlib install — never raises, so a
    ground-truth mask rasterized for scoring can't crash a run.

    :func:`_rasterize_object` (CityRefer contours) routes through here —
    one rasterizer, one cell-index convention.
    """
    H, W = grid_shape
    if not polygon or len(polygon) < 3:
        return []
    try:
        from matplotlib.path import Path as MplPath
    except ImportError:
        return []

    poly = MplPath(np.asarray(polygon, dtype=float))
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    col_min = max(0, int((min(xs) - bounds.x_min) / cell_size_m))
    col_max = min(W - 1, int((max(xs) - bounds.x_min) / cell_size_m))
    row_min = max(0, int((min(ys) - bounds.y_min) / cell_size_m))
    row_max = min(H - 1, int((max(ys) - bounds.y_min) / cell_size_m))
    if col_max < col_min or row_max < row_min:
        return []
    cc, rr = np.meshgrid(
        np.arange(col_min, col_max + 1),
        np.arange(row_min, row_max + 1),
    )
    centers_x = bounds.x_min + (cc + 0.5) * cell_size_m
    centers_y = bounds.y_min + (rr + 0.5) * cell_size_m
    pts = np.stack([centers_x.ravel(), centers_y.ravel()], axis=1)
    inside = poly.contains_points(pts).reshape(rr.shape)
    return [[int(r), int(c)] for r, c in zip(rr[inside], cc[inside])]


def _rasterize_object(
    obj: CityRefObject,
    bounds: WorldBounds,
    grid_shape: tuple[int, int],
    cell_size_m: float,
) -> list[list[int]]:
    """Rasterize an object's footprint into ``[[row, col], ...]`` cells.

    Uses the 2D contour when available (via :func:`rasterize_polygon`), else
    falls back to the axis-aligned bbox from position+dimension.
    """
    H, W = grid_shape

    if obj.contour and len(obj.contour) >= 3:
        cells = rasterize_polygon(obj.contour, bounds, grid_shape, cell_size_m)
        if cells:
            return cells

    # Bbox fallback.
    cx, cy, _ = obj.position
    dx, dy, _ = obj.dimension
    x0, x1 = cx - dx / 2.0, cx + dx / 2.0
    y0, y1 = cy - dy / 2.0, cy + dy / 2.0
    col_min = max(0, int((x0 - bounds.x_min) / cell_size_m))
    col_max = min(W - 1, int((x1 - bounds.x_min) / cell_size_m))
    row_min = max(0, int((y0 - bounds.y_min) / cell_size_m))
    row_max = min(H - 1, int((y1 - bounds.y_min) / cell_size_m))
    # Always occupy at least the center cell so tiny objects don't vanish.
    if col_max < col_min or row_max < row_min:
        col = max(0, min(W - 1, int((cx - bounds.x_min) / cell_size_m)))
        row = max(0, min(H - 1, int((cy - bounds.y_min) / cell_size_m)))
        return [[row, col]]
    for r in range(row_min, row_max + 1):
        for c in range(col_min, col_max + 1):
            cells.append([r, c])
    return cells
