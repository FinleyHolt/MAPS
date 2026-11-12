"""A bake's local metres to WGS84 and back, off its own ``<map>_meta.json`` georef.

A map with no ``georef`` block converts nothing: the functions answer None
rather than a guess, so a caller can tell "this map cannot be placed" from a
coordinate.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

#: The ground's orthometric height under each bake, metres MSL: what the
#: receiver's MSL altitude is measured against for the height above ground.
#: Cal City: the 3DEP median over the block.
FLOOR_MSL_M: dict[str, float] = {"calcity": 722.8}


class Bounds(NamedTuple):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def map_bounds(tiles_dir: str | Path, map_name: str) -> Bounds:
    """The bake's extent in its own metres, off ``<map>_meta.json``."""
    b = json.loads((Path(tiles_dir) / f"{map_name}_meta.json").read_text())["bounds"]
    return Bounds(float(b["x_min"]), float(b["y_min"]), float(b["x_max"]), float(b["y_max"]))


@lru_cache(maxsize=8)
def _transformers(tiles_dir: str, map_name: str):
    """``(to_lonlat, to_utm, origin_e, origin_n)`` for the bake, or None without a georef."""
    from pyproj import Transformer

    path = Path(tiles_dir) / f"{map_name}_meta.json"
    try:
        georef = json.loads(path.read_text()).get("georef") or {}
    except (OSError, ValueError):
        return None
    epsg, e0, n0 = georef.get("crs_utm_epsg"), georef.get("origin_e"), georef.get("origin_n")
    if epsg is None or e0 is None or n0 is None:
        return None
    return (
        Transformer.from_crs(f"EPSG:{int(epsg)}", "EPSG:4326", always_xy=True),
        Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}", always_xy=True),
        float(e0),
        float(n0),
    )


def local_to_lonlat(tiles_dir, map_name: str, x: float, y: float) -> tuple[float, float] | None:
    """``(lon, lat)`` of a local ``(x, y)``, or None on a map with no georef."""
    t = _transformers(str(tiles_dir), map_name)
    if t is None:
        return None
    lon, lat = t[0].transform(x + t[2], y + t[3])
    return float(lon), float(lat)


def lonlat_to_local(tiles_dir, map_name: str, lon: float, lat: float) -> tuple[float, float] | None:
    """Local ``(x, y)`` of a WGS84 ``(lon, lat)``, or None on a map with no georef."""
    t = _transformers(str(tiles_dir), map_name)
    if t is None:
        return None
    e, n = t[1].transform(lon, lat)
    return float(e - t[2]), float(n - t[3])
