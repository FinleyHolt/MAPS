#!/usr/bin/env bash
# Bake the CityNav tile sidecars into tiles/citynav_2m_osmv3 from the shipped
# OpenStreetMap caches (tiles/osm_cache). Every bake stamps meta.json with its
# provenance (cache name + sha256, georef residuals, bake time + git sha).
#
# Usage (from the repo root):
#   PY=.venv/bin/python benchmarks/citynav/data_prep/bake_tiles.sh
#   TILES_DIR=tiles/citynav_2m_osmv3 benchmarks/citynav/data_prep/bake_tiles.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

TILES_DIR="${TILES_DIR:-${SEARCH_CITYNAV_TILES:-tiles/citynav_2m_osmv3}}"
PY="${PY:-python3}"
# The bake derives block bounds AND imagery coverage from the rasters. A
# python without rasterio/scipy does not crash: it falls back to the sidecar
# box and rectangle coverage and completes full of priors nothing can
# observe. Fail loud instead.
if ! "$PY" -c 'import rasterio, scipy' >/dev/null 2>&1; then
	echo "error: ${PY} lacks rasterio/scipy; use the repo env, e.g. PY=.venv/bin/python $0" >&2
	exit 3
fi

for CITY in birmingham cambridge; do
	"$PY" -m benchmarks.citynav.bake.city --city "$CITY" \
		--osm-cache "tiles/osm_cache/${CITY}_osm_local.geojson" \
		--tiles-dir "$TILES_DIR"
done

echo "Bake complete: $TILES_DIR"
echo "Point the runner at it: export SEARCH_CITYNAV_TILES=$TILES_DIR"
