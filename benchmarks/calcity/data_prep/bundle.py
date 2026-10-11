"""Read a world bundle as the Cal City data source.

A world bundle is plain files: ``osm/<layer>.geojson`` in WGS84, one per
``OSM_LAYERS`` entry. This module is the only place Cal City touches that
layout. The block's imagery is not in the bundle: it is the block's own map,
``data/rgbd_upstream/<map>_rgb.tif``.

Two consumers:

* :func:`load_features_local` — every OSM feature across all vector layers,
  geometry reprojected into a block's local CityNav task frame as
  shapely objects, tags verbatim.
* :func:`grid_shape` — the block raster's ``(H, W)`` at a cell, from the frame.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator


from benchmarks.calcity.data_prep.aoi import LocalFrame

log = logging.getLogger("calcity.bundle")

# The vector layers a world bundle carries; a missing one is refused.
OSM_LAYERS = (
    "buildings",
    "roads",
    "aeroways",
    "water",
    "landuse",
    "barriers",
    "power",
    "pois",
)


def iter_features(bundle: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(layer_name, geojson_feature)`` across every vector layer."""
    for layer in OSM_LAYERS:
        path = bundle / "osm" / f"{layer}.geojson"
        if not path.is_file():
            raise FileNotFoundError(f"the world bundle has no {path}")
        fc = json.loads(path.read_text())
        for feature in fc.get("features", []):
            yield layer, feature


def load_features_local(bundle: Path, lf: LocalFrame) -> list[dict[str, Any]]:
    """All bundle features intersecting the block, in the local frame.

    Returns records ``{"layer", "osm_id", "tags", "geom"}`` where ``geom`` is
    a shapely geometry in local meters. Features wholly outside the block are
    dropped; features straddling the boundary are kept un-clipped (a landmark
    on the seam is still a valid whole-shape target, mirroring how CityNav
    blocks window a shared city frame).
    """
    from pyproj import Transformer
    from shapely.geometry import shape, box
    from shapely.ops import transform as shp_transform

    t = Transformer.from_crs("EPSG:4326", lf.crs_utm, always_xy=True)

    def to_local(x, y, z=None):
        e, n = t.transform(x, y)
        return lf.utm_to_runtime(e, n)

    window = box(*lf.bounds_local)
    records: list[dict[str, Any]] = []
    for layer, feature in iter_features(bundle):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        geom = shp_transform(to_local, shape(geometry))
        if not geom.intersects(window):
            continue
        records.append(
            {
                "layer": layer,
                "osm_id": feature.get("id"),
                "tags": dict(feature.get("properties") or {}),
                "geom": geom,
            }
        )
    log.info("bundle: %d features intersect the block window", len(records))
    return records


def grid_shape(lf: LocalFrame, m_per_px: float) -> tuple[int, int]:
    """``(H, W)`` of the block raster at ``m_per_px``."""
    return int(round(lf.height_m / m_per_px)), int(round(lf.width_m / m_per_px))


