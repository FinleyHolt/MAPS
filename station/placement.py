"""Where a map sits on the earth, for display: its local metres to WGS84 and back.

A bake that carries its own georef (Cal City's ``<map>_meta.json``) is placed
by it (``benchmarks.flight.tile_georef``). A CityNav bake carries none, and
its files are not the station's to change; the city's shipped OSM cache
(``tiles/osm_cache/<city>_osm_local.geojson``) states the fitted transform
its features were brought into CityNav local XY with, ``osm_crs`` and
``citynav_from_osm_affine``, so a block of that city is placed by inverting
it. A map with neither is placed nowhere: :func:`placement` answers None, and
a caller draws nothing rather than guess.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from benchmarks import REPO_ROOT
from benchmarks.flight import tile_georef

#: The shipped OSM caches, one per CityNav city.
OSM_CACHE_DIR = REPO_ROOT / "tiles" / "osm_cache"


class Placement(NamedTuple):
    #: ``(x, y)`` local metres -> ``(lon, lat)``.
    to_lonlat: Callable[[float, float], tuple[float, float]]
    #: ``(lon, lat)`` -> ``(x, y)`` local metres.
    to_local: Callable[[float, float], tuple[float, float]]


@lru_cache(maxsize=8)
def _city(osm_cache_dir: str, city: str) -> Placement | None:
    """The placement a city's OSM cache states, or None without a cache that states one."""
    from pyproj import Transformer  # noqa: PLC0415

    try:
        payload = json.loads((Path(osm_cache_dir) / f"{city}_osm_local.geojson").read_text())
        georef = payload["citynav_osm_cache"]["georef"]
        # local = [[a, b], [d, e]] @ (east, north) + (c, f)
        a, b, c, d, e, f = (float(v) for v in georef["citynav_from_osm_affine"])
        crs = str(georef["osm_crs"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    det = a * e - b * d
    if not det:
        return None
    out = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    back = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def to_lonlat(x: float, y: float) -> tuple[float, float]:
        u, v = x - c, y - f
        lon, lat = out.transform((e * u - b * v) / det, (a * v - d * u) / det)
        return float(lon), float(lat)

    def to_local(lon: float, lat: float) -> tuple[float, float]:
        east, north = back.transform(lon, lat)
        return float(a * east + b * north + c), float(d * east + e * north + f)

    return Placement(to_lonlat, to_local)


@lru_cache(maxsize=64)
def placement(tiles_dir: str, osm_cache_dir: str, map_name: str) -> Placement | None:
    """How ``map_name`` is placed: by its bake's georef, else by its city's OSM cache, else None."""
    if tile_georef.local_to_lonlat(tiles_dir, map_name, 0.0, 0.0) is not None:
        return Placement(lambda x, y: tile_georef.local_to_lonlat(tiles_dir, map_name, x, y),
                         lambda lon, lat: tile_georef.lonlat_to_local(tiles_dir, map_name, lon, lat))
    # A CityNav block is ``<city>_block_<n>``, in its city's one local frame.
    city, sep, _ = map_name.partition("_block_")
    return _city(osm_cache_dir, city) if sep else None
