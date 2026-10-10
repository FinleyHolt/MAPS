#!/usr/bin/env python3
"""Bake Cal City's OSM prior with the streamlined CityNav baker.

The bundle's OSM goes STRAIGHT to the baker: reprojected into the AOI's local
metric frame, tags verbatim, geometry kind intact, nothing classified and
nothing dropped. That is the same contract CityNav bakes under -- see
``benchmarks.citynav.bake.city._read_geojson_cache``, "Every feature with a
geometry is kept -- no classification, no drop-gate."

``objects.json``, the CityRefer catalog, is the ground-truth and authoring
substrate (episode ``object_ids``); the prior is baked from the bundle alone.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.calcity import paths
from benchmarks.calcity.data_prep.aoi import (
    BLOCKS,
    CELL_SIZE_M,
    build_local_frame,
    bundle_root,
)
from benchmarks.calcity.data_prep.bundle import load_features_local
from benchmarks.citynav.bake.city import bake_city_from_cache


def _write_meta(map_name: str, tiles_dir: Path) -> Path:
    frame = build_local_frame(BLOCKS[map_name])
    width = round(frame.width_m / CELL_SIZE_M)
    height = round(frame.height_m / CELL_SIZE_M)
    payload = {
        "block": map_name,
        "grid_shape": [height, width],
        "cell_size_m": CELL_SIZE_M,
        "coordinate_frame": {
            "x_axis": "east",
            "y_axis": "north",
            "origin": "southwest",
        },
        "bounds": {
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": round(frame.width_m, 4),
            "y_max": round(frame.height_m, 4),
        },
        "georef": {
            "crs_utm_epsg": int(frame.crs_utm.to_epsg()),
            "origin_e": round(frame.origin_e, 4),
            "origin_n": round(frame.origin_n, 4),
            "runtime_transform": "x=easting-origin_e; y=northing-origin_n",
            "aoi_bbox_wgs84": list(BLOCKS[map_name]),
        },
    }
    tiles_dir.mkdir(parents=True, exist_ok=True)
    path = tiles_dir / f"{map_name}_meta.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def local_osm_cache(bundle: Path, map_name: str) -> dict[str, Any]:
    """The bundle's OSM as a GeoJSON cache in the map's local metric frame.

    The ONLY transform is the reprojection. Tags ride verbatim, every geometry
    kind survives as itself (a road stays a LineString), and no feature is
    classified or dropped -- ``load_features_local`` already keeps a
    boundary-straddling feature un-clipped, and the baker clips cells to the
    grid on its own.

    The layer a feature came from rides as ``worldkit_layer``. OSM's own
    ``source`` tag is left alone: the old builder overwrote it, which is a
    tag the map actually carries.
    """
    from shapely.geometry import mapping

    lf = build_local_frame(BLOCKS[map_name])
    records = load_features_local(bundle, lf)
    features: list[dict[str, Any]] = []
    for rec in records:
        geom = rec.get("geom")
        if geom is None or geom.is_empty:
            continue
        props = {str(k): v for k, v in (rec.get("tags") or {}).items()}
        props["worldkit_layer"] = rec.get("layer")
        features.append(
            {
                "type": "Feature",
                "id": rec.get("osm_id"),
                "properties": props,
                "geometry": mapping(geom),
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "citynav_osm_cache": {
            "source": "calcity worldkit bundle OSM",
            "records": len(features),
            "map_name": map_name,
        },
    }


def bake_block(bundle: Path, tiles_dir: Path, map_name: str) -> dict[str, Any]:
    cache = local_osm_cache(bundle, map_name)
    if not cache["features"]:
        raise ValueError(f"{bundle} carries no OSM features over {map_name!r}")
    _write_meta(map_name, tiles_dir)
    with tempfile.TemporaryDirectory(prefix=f"{map_name}_osm_") as tmp:
        cache_path = Path(tmp) / f"{map_name}_osm_local.geojson"
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
        return bake_city_from_cache(map_name, cache_path, tiles_dir=tiles_dir, imagery=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--tiles-dir", type=Path, default=paths.TILES_DIR)
    parser.add_argument(
        "--blocks", nargs="+", default=sorted(BLOCKS), choices=sorted(BLOCKS)
    )
    args = parser.parse_args(argv)
    bundle = args.bundle or bundle_root()
    for map_name in args.blocks:
        stats = bake_block(bundle, args.tiles_dir, map_name)
        print(
            f"[calcity-bake] {map_name}: "
            f"{stats['features']} priors, {stats['nodes']} graph nodes"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
