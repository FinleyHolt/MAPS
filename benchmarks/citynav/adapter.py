"""CityNav episodes and CityRefer objects, as ``citynav-arxiv/gsamllavanav/dataset`` lays them out.

A split JSON (``data/citynav/citynav_<split>.json``) is a list of MTurk
trajectory rows: ``area``, ``block``, ``descriptions``, ``target_positions``,
``marker_positions``, ``object_ids``, ``ann_ids``, ``trajectory``.
``data/cityrefer/objects.json`` maps a map name to objects carrying ``id``,
``name``, ``object_type``, ``position`` (xyz centre), ``dimension`` and
``contour`` (2D polygon): the scorer's answer key.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


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
    contour: list[tuple[float, float]] = field(default_factory=list)  # 2D polygon


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
    #: Upstream's start yaw as this tree's heading (degrees clockwise from
    #: north, :func:`start_heading_deg`). The replay starts facing it.
    start_heading_deg: float = 0.0
    success_radius_m: float | None = None
    """The row's own fallback success radius (Cal City publishes a 20 m and a
    precise dataset); ``None`` leaves the run config's in charge."""


# ---------------------------------------------------------------------------
# Split + objects loading
# ---------------------------------------------------------------------------


def load_split(split_file: Path) -> list[dict]:
    """Load a CityNav split JSON (list of MTurk trajectory rows)."""
    with open(split_file) as f:
        return json.load(f)


def load_objects(objects_json: Path) -> dict[str, list[CityRefObject]]:
    """``cityrefer/objects.json`` keyed by map name, from a nested
    ``{map_name: {object_id: {...}}}`` dict or a flat list."""
    with open(objects_json) as f:
        raw = json.load(f)

    by_map: dict[str, list[CityRefObject]] = {}

    def _mk(map_name: str, obj: dict) -> CityRefObject | None:
        try:
            pos = obj["position"]
            obj["dimension"]  # a CityRefer object carries one; a row without is refused
            contour_raw = obj.get("contour", []) or []
            return CityRefObject(
                map_name=map_name,
                id=int(obj["id"]),
                name=str(obj.get("name") or obj.get("object_name") or ""),
                object_type=str(obj.get("object_type") or obj.get("type") or ""),
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                contour=[(float(p[0]), float(p[1])) for p in contour_raw],
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
    """``<area>_block_<block>``; ``area`` verbatim when it already contains
    "block" or the row has no ``block`` (a whole-AOI map)."""
    area = str(row["area"])
    block = row.get("block")
    block_s = "" if block is None else str(block)
    return f"{area}_block_{block_s}" if (block_s and "block" not in area) else area


def start_heading_deg(start_pose: list) -> float:
    """The start heading of ``trajectory[0]`` as upstream's ``MTurkTrajectory`` reads it.

    ``[x, y, z, yaw, pitch]`` carries the yaw, ``[x, y, z, dx, dy, dz]`` the
    yaw ``atan2(dy, dx)`` (radians counter-clockwise from east); ``[x, y, z]``
    (Cal City) faces north; any other length raises. Returned in degrees
    clockwise from north, in ``[0, 360)``.
    """
    if len(start_pose) == 3:
        return 0.0
    if len(start_pose) == 5:
        yaw = float(start_pose[3])
    elif len(start_pose) == 6:
        yaw = math.atan2(float(start_pose[4]), float(start_pose[3]))
    else:
        raise ValueError(
            f"trajectory[0] has {len(start_pose)} values; upstream reads "
            "[x, y, z, yaw, pitch] or [x, y, z, dx, dy, dz]"
        )
    return (90.0 - math.degrees(yaw)) % 360.0


def parse_episode(row: dict, row_index: int | None = None) -> CityNavEpisode:
    """One split-JSON row as a CityNavEpisode, id ``<map>::<ann>#<row_index>``.

    Rows of one block share ``ann_ids[0]`` (~99% of ``map::ann`` ids collide),
    so ``row_index`` makes the id unique; without it the id is ``map::ann``.
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
        start_heading_deg=start_heading_deg(trajectory[0]),
        target_xy=(tx, ty),
        target_object_ids=list(row.get("object_ids") or []),
        target_mask=target_mask,
        marker_positions=markers,
        success_radius_m=(
            float(_raw_radius) if _raw_radius is not None else None
        ),
    )


def find_row(rows: list[dict], episode_id: str) -> tuple[int, dict] | None:
    """``(index, row)`` of the split row ``episode_id`` names, else None:
    ``<map>::<ann>#<i>`` is row ``i`` when its bare id matches, a bare
    ``<map>::<ann>`` the first row carrying it."""
    base, sep, suffix = episode_id.rpartition("#")
    if sep:
        try:
            i = int(suffix)
        except ValueError:
            return None
        if 0 <= i < len(rows) and parse_episode(rows[i]).episode_id == base:
            return i, rows[i]
        return None
    return next(((i, r) for i, r in enumerate(rows) if parse_episode(r).episode_id == episode_id), None)
