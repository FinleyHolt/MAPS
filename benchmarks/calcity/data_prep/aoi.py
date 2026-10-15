"""Shared AOI definitions + local metric frame for the Cal City map.

Cal City is ONE map: the California City, CA area of operations, which
holds the town's developed core.

Runtime frame
-------------
WGS84 -> UTM (zone 11N / EPSG:32611), then map into the CityNav task
frame: local ``x`` increases east and local ``y`` increases north — the
engine's one world frame. The south-west corner is ``(0, 0)`` and the
north-east corner is ``(width_m, height_m)``, so geographic compass words
in Cal City instructions align with the runtime's cardinals with no
transform at this boundary.

Terrain / flight regime
-----------------------
The world is FLAT by construction: the DSM raster is a constant plane at
``GROUND_LEVEL_M`` (0.0), so ``pose.z`` reads directly as height above ground
and the ortho view square (side = ``2*(z - GROUND_LEVEL[map])``) sees the
whole altitude band with no terrain coupling. California City's core sits on
gently-sloped desert floor (~30 m of true relief across 6 km), which the 2D
imagery cannot resolve anyway; a constant plane is the honest encoding of a
2D-imagery world.

World source
------------
OSM vectors come from a world bundle (``osm/<layer>.geojson``), located
via ``CALCITY_BUNDLE`` (default ``bundle/`` under the Cal City root); the imagery
is the program's own map of the block, ``data/rgbd_upstream/<map>_rgb.tif``.
The bundle must cover ``BUNDLE_BBOX_WGS84``, which is slightly larger than the map bbox
because a lon/lat rectangle is not axis-aligned after projection to UTM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

# The one map, ``calcity``, as a WGS84 bbox (w, s, e, n): the briefed area of
# operations plus 3 km on every side, so a C-12 hold about a target at the
# AO's edge, or a turn back into it, stays on imagery. Rows carry
# ``area: "calcity"`` and no block key.
AOI_BBOX: tuple[float, float, float, float] = (-118.0366, 35.0763, -117.89, 35.1672)

# Fetch extent for the world bundle. The ~130 m padding covers
# the UTM grid-convergence bulge at the map edges; fetching only the map bbox
# leaves black pixels along parts of the projected canvas.
BUNDLE_BBOX_WGS84: tuple[float, float, float, float] = (
    -118.038,
    35.0751,
    -117.8886,
    35.1684,
)

# Flat-world ground plane (see module docstring), the DSM constant;
# ``GROUND_LEVEL["calcity"]`` carries the same value.
GROUND_LEVEL_M = 0.0

def bundle_root() -> Path:
    """The world bundle directory (``CALCITY_BUNDLE`` env override), else
    ``bundle/`` under the Cal City root."""
    from benchmarks.calcity.paths import CALCITY_ROOT  # noqa: PLC0415

    return Path(os.environ.get("CALCITY_BUNDLE", str(CALCITY_ROOT / "bundle"))).expanduser()


@dataclass(frozen=True)
class LocalFrame:
    """UTM envelope plus Cal City's CityNav-compatible runtime transform."""

    crs_utm: object  # pyproj.CRS
    origin_e: float  # UTM easting of the AOI west edge (subtracted -> local x)
    origin_n: float  # UTM northing of the AOI south edge (subtracted -> local y)
    width_m: float
    height_m: float

    @property
    def bounds_local(self) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) of the AOI in the local frame."""
        return (0.0, 0.0, self.width_m, self.height_m)

    def utm_to_runtime(self, easting, northing):
        """Map UTM coordinates to runtime ``(x east, y north)`` meters.

        Scalars and numpy-like arrays are both accepted so this is also safe
        to pass through ``shapely.ops.transform``.
        """
        return (
            easting - self.origin_e,
            northing - self.origin_n,
        )


def build_local_frame(bbox: tuple[float, float, float, float]) -> LocalFrame:
    """Construct a block's local metric frame from its WGS84 bbox corners."""
    from pyproj import CRS, Transformer

    w, s, e, n = bbox
    lon_c = (w + e) / 2.0
    lat_c = (s + n) / 2.0
    zone = int((lon_c + 180) / 6) + 1
    hemisphere = "north" if lat_c >= 0 else "south"
    crs_utm = CRS(proj="utm", zone=zone, datum="WGS84", **{hemisphere: True})

    t = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    # Transform all four corners; the lon/lat rectangle is not axis-aligned in
    # UTM, so take the min/max easting & northing for an axis-aligned UTM bbox.
    corners = [(w, s), (e, s), (e, n), (w, n)]
    es, ns = zip(*(t.transform(lon, lat) for lon, lat in corners))
    origin_e, origin_n = min(es), min(ns)
    width_m = max(es) - origin_e
    height_m = max(ns) - origin_n

    return LocalFrame(
        crs_utm=crs_utm,
        origin_e=float(origin_e),
        origin_n=float(origin_n),
        width_m=float(width_m),
        height_m=float(height_m),
    )


# --------------------------------------------------------------------------
# OSM tag -> CityRefer object_type classification
# --------------------------------------------------------------------------
#
# ``object_type`` is one of the 13 SensatUrban / CityRefer classes; the
# objects.json build keeps only classified footprints and ranks co-located
# duplicates by it. The rules read buildings, roads, landuse, water and aeroway
# (the airport corner).

_LEISURE_WATER = {"swimming_pool"}
_LEISURE_GROUND = {"pitch", "track", "garden", "stadium", "bleachers", "park", "golf_course"}
_AEROWAY_BUILDING = {"hangar", "terminal"}
_AEROWAY_GROUND = {"runway", "taxiway", "apron", "helipad", "runway_link"}
_HIGHWAY_DRIVE = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}
_HIGHWAY_FOOT = {"footway", "pedestrian", "path", "steps", "cycleway"}
_VEG_NATURAL = {"wood", "scrub", "heath", "grassland", "tree_row"}
_VEG_LANDUSE = {"forest", "grass", "meadow", "recreation_ground", "village_green"}

_MISSING = frozenset({"", "<NA>", "nan", "none", "null"})


def tag_value(tags: dict, key: str) -> str | None:
    """Return ``tags[key]`` as a clean string, or ``None`` when absent/NA."""
    v = tags.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in _MISSING else s


def classify_object_type(tags: dict) -> str | None:
    """Map an OSM feature's tags to a CityRefer ``object_type`` (or ``None``).

    ``None`` means "skip this feature".
    """
    leisure = tag_value(tags, "leisure")
    natural = tag_value(tags, "natural")
    amenity = tag_value(tags, "amenity")
    highway = tag_value(tags, "highway")
    aeroway = tag_value(tags, "aeroway")
    landuse = tag_value(tags, "landuse")
    water = tag_value(tags, "water")
    name = tag_value(tags, "name")

    if natural == "water" or water is not None or leisure in _LEISURE_WATER:
        return "Water"
    # A real building footprint wins over a co-tagged surface so plaza /
    # hangar / church shells are targetable as structures.
    if tag_value(tags, "building") is not None or aeroway in _AEROWAY_BUILDING:
        return "Building"
    if aeroway in _AEROWAY_GROUND:
        return "Ground"
    if leisure in _LEISURE_GROUND or leisure == "sports_center":
        return "Ground"
    if landuse == "golf_course" or leisure == "golf_course":
        return "Ground"
    if amenity == "parking":
        return "Parking"
    # Business/civic POIs (nodes or areas): what the imagery shows at a
    # restaurant / bank / market / motel is a building, so they class as
    # Building and stay targetable by name.
    if amenity is not None or tag_value(tags, "shop") is not None or tag_value(
        tags, "tourism"
    ) is not None:
        return "Building"
    # Named plazas / commercial lots are open targetable areas.
    if landuse in ("retail", "commercial") and name:
        return "Ground"
    if highway in _HIGHWAY_DRIVE:
        return "TrafficRoad"
    if highway == "service" and name:
        return "TrafficRoad"
    if highway in _HIGHWAY_FOOT and name:
        return "Footpath"
    if natural in _VEG_NATURAL or landuse in _VEG_LANDUSE:
        return "HighVegetation"
    return None
