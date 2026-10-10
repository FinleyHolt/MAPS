#!/usr/bin/env bash
# Bake the CityNav tile sidecars into tiles/citynav_2m_osmv3: the OSM prior
# (raw internet OSM, tags verbatim). Nothing is redistributed; everything
# regenerates from the fetched data (data_prep/fetch_data.sh first). Every
# bake stamps meta.json with its provenance (source cache + sha256 + fetch
# time, georef residuals, bake time + git sha).
#
# Per city:
#   a. fetch OSM once into a local-XY cache (network; georef shipped in
#      benchmarks/citynav/georef/; delete the cache for a fresh pull)
#   b. offline re-bake of every block from the cache
#
# Usage (from the repo root, in the env with the `bake` extra installed):
#   benchmarks/citynav/data_prep/bake_tiles.sh
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
mkdir -p "$TILES_DIR" tiles/osm_cache

for CITY in birmingham cambridge; do
	CACHE="tiles/osm_cache/${CITY}_osm_local.geojson"
	if [[ ! -f "$CACHE" ]]; then
		echo "fetching OSM for ${CITY} (network, once) -> ${CACHE}"
		"$PY" -m benchmarks.citynav.bake.city \
			--dump-osm-cache "$CACHE" \
			--georef "benchmarks/citynav/georef/${CITY}_georef.json" \
			--city "$CITY"
	else
		echo "reusing OSM cache ${CACHE} (delete it to force a fresh pull)"
	fi
	echo "offline bake of every ${CITY} block"
	"$PY" -m benchmarks.citynav.bake.city \
		--osm-blocks --city "$CITY" \
		--osm-cache "$CACHE" \
		--tiles-dir "$TILES_DIR"
done

echo "Bake complete: $TILES_DIR"
echo "Point the runner at it: export SEARCH_CITYNAV_TILES=$TILES_DIR"
