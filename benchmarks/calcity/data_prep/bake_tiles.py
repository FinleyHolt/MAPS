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
from pathlib import Path
from typing import Any

from benchmarks.calcity import paths
from benchmarks.calcity.data_prep.aoi import (
    AOI_BBOX,
    CELL_SIZE_M,
    build_local_frame,
    bundle_root,
)
from benchmarks.calcity.data_prep.bundle import load_features_local
from benchmarks.citynav.bake import city
from benchmarks.citynav.bake.city import _normalise_props


def _write_meta(map_name: str, tiles_dir: Path) -> Path:
    frame = build_local_frame(AOI_BBOX)
    width = round(frame.width_m / CELL_SIZE_M)
    height = round(frame.height_m / CELL_SIZE_M)
    payload = {
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
        },
    }
    tiles_dir.mkdir(parents=True, exist_ok=True)
    path = tiles_dir / f"{map_name}_meta.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def local_osm_records(bundle: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The bundle's OSM as baker records in the map's local metric frame.

    The ONLY transform is the reprojection. Tags ride verbatim, every geometry
    kind survives as itself (a road stays a LineString), and no feature is
    classified or dropped -- ``load_features_local`` already keeps a
    boundary-straddling feature un-clipped, and the baker clips cells to the
    grid on its own.

    The layer a feature came from rides as ``worldkit_layer``; OSM's own
    ``source`` tag rides as ``source_tag``, as in every bake.
    """
    from shapely.geometry import mapping

    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for rec in load_features_local(bundle, build_local_frame(AOI_BBOX)):
        geom = rec.get("geom")
        if geom is None or geom.is_empty:
            continue
        tags = {str(k): v for k, v in (rec.get("tags") or {}).items()}
        props = _normalise_props({**tags, "worldkit_layer": rec.get("layer")})
        if rec.get("osm_id") is not None:
            props.setdefault("osm_id", rec["osm_id"])
        records.append((props, mapping(geom)))
    return records


def bake_block(bundle: Path, tiles_dir: Path, map_name: str) -> dict[str, Any]:
    records = local_osm_records(bundle)
    if not records:
        raise ValueError(f"{bundle} carries no OSM features over {map_name!r}")
    _write_meta(map_name, tiles_dir)
    return city.bake_block(map_name, records, tiles_dir, imagery=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--tiles-dir", type=Path, default=paths.TILES_DIR)
    args = parser.parse_args(argv)
    stats = bake_block(args.bundle or bundle_root(), args.tiles_dir, "calcity")
    print(f"[calcity-bake] calcity: {stats['features']} priors, {stats['nodes']} graph nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
