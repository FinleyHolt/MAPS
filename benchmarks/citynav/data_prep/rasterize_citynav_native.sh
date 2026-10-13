#!/usr/bin/env bash
# Generate the upstream-faithful CityNav nadir rasters the ortho observation
# reads, by running upstream's exact
# PDAL/GDAL `rasterize.py` (vendored beside this script as
# _vendor_citynav_rasterize.py)
# on the SensatUrban block PLYs.
#
# Upstream ships NO downloadable prebaked rgbd: its README has every user
# run this same deterministic script on the SensatUrban point clouds. The
# script IS the canonical artifact, so this output is byte-faithful to any
# upstream user's data/rgbd (0.1 m/px, GDAL max-Z height + mean-RGB; corrupt
# cambridge blocks 0/1/34 skipped, matching upstream).
#
# Default output: $CITYNAV_ROOT/data/rgbd_upstream/ (override with
# $TARGET_DIR or the runtime env var SEARCH_CITYNAV_RGBD_DIR).
#
# Manual one-time setup step (~minutes/PLY in PDAL); not run by the harness.
# Runs in a pre-existing `rasterize` env (pdal + rasterio + pillow),
# exactly upstream's scripts/rasterize.sh environment. Override the
# interpreter with RASTERIZE_PYTHON=/path/to/env/bin/python.
set -euo pipefail

CITYNAV_ROOT="${SEARCH_CITYNAV_DATA:-${CITYNAV_ROOT:-$HOME/datasets/citynav-arxiv}}"
TARGET_DIR="${TARGET_DIR:-${SEARCH_CITYNAV_RGBD_DIR:-$CITYNAV_ROOT/data/rgbd_upstream}}"
RASTERIZE_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_vendor_citynav_rasterize.py"
RASTERIZE_ENV_NAME="${RASTERIZE_ENV_NAME:-rasterize}"
RASTERIZE_PYTHON="${RASTERIZE_PYTHON:-}"
BLOCK_FILTERS=()

while [[ $# -gt 0 ]]; do
	case "$1" in
	--blocks)
		shift
		before="${#BLOCK_FILTERS[@]}"
		while [[ $# -gt 0 && "$1" != --* ]]; do
			BLOCK_FILTERS+=("$1")
			shift
		done
		if [[ "${#BLOCK_FILTERS[@]}" -eq "$before" ]]; then
			echo "[citynav-rasterize] ERROR: --blocks requires at least one block name" >&2
			exit 2
		fi
		;;
	--blocks=*)
		BLOCK_FILTERS+=("${1#--blocks=}")
		shift
		;;
	*)
		echo "[citynav-rasterize] ERROR: unknown argument: $1" >&2
		exit 2
		;;
	esac
done

if [[ -n "$RASTERIZE_PYTHON" ]]; then
	if [[ ! -x "$RASTERIZE_PYTHON" ]]; then
		echo "[citynav-rasterize] ERROR: rasterize python not executable: $RASTERIZE_PYTHON" >&2
		exit 1
	fi
	RASTERIZE_CMD=("$RASTERIZE_PYTHON")
elif [[ -x "$HOME/micromamba/envs/${RASTERIZE_ENV_NAME}/bin/python" ]]; then
	RASTERIZE_CMD=("$HOME/micromamba/envs/${RASTERIZE_ENV_NAME}/bin/python")
elif [[ -x "$HOME/.local/share/mamba/envs/${RASTERIZE_ENV_NAME}/bin/python" ]]; then
	RASTERIZE_CMD=("$HOME/.local/share/mamba/envs/${RASTERIZE_ENV_NAME}/bin/python")
elif command -v micromamba >/dev/null 2>&1; then
	RASTERIZE_CMD=("micromamba" "run" "-n" "$RASTERIZE_ENV_NAME" "python")
else
	echo "[citynav-rasterize] ERROR: rasterize env python not found." >&2
	echo "[citynav-rasterize] Set RASTERIZE_PYTHON=/path/to/rasterize/bin/python," >&2
	echo "[citynav-rasterize] or create the upstream env:" >&2
	echo "[citynav-rasterize]   micromamba create -n rasterize -c conda-forge pdal python=3.10" >&2
	echo "[citynav-rasterize]   micromamba run -n rasterize pip install rasterio pillow" >&2
	exit 1
fi

# Upstream runs rasterize.py once per PLY split dir.
PLY_DIRS=()
TOTAL_PLYS=0
for d in "$CITYNAV_ROOT/raw_ply/train" "$CITYNAV_ROOT/raw_ply/test"; do
	[[ -d "$d" ]] || continue
	if [[ "${#BLOCK_FILTERS[@]}" -gt 0 ]]; then
		n=0
		for block in "${BLOCK_FILTERS[@]}"; do
			if [[ -f "$d/$block.ply" ]]; then
				n=$((n + 1))
			fi
		done
	else
		n="$(find "$d" -maxdepth 1 -name '*.ply' | wc -l)"
	fi
	if [[ "$n" -gt 0 ]]; then
		PLY_DIRS+=("$d")
		TOTAL_PLYS=$((TOTAL_PLYS + n))
	fi
done

if [[ "$TOTAL_PLYS" -eq 0 ]]; then
	if [[ "${#BLOCK_FILTERS[@]}" -gt 0 ]]; then
		echo "[citynav-rasterize] ERROR: requested block PLYs not found: ${BLOCK_FILTERS[*]}" >&2
	else
		echo "[citynav-rasterize] ERROR: no SensatUrban .ply files found under:" >&2
	fi
	echo "[citynav-rasterize]   $CITYNAV_ROOT/raw_ply/{train,test}" >&2
	echo "[citynav-rasterize] Fetch the SensatUrban PLYs, then rerun this script." >&2
	exit 1
fi

mkdir -p "$TARGET_DIR"
# rasterize.py writes temp z/r/g/b.tif into CWD, so run from a scratch dir.
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/citynav_rasterize.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

echo "[citynav-rasterize] env=${RASTERIZE_CMD[*]}"
echo "[citynav-rasterize] target=$TARGET_DIR"
if [[ "${#BLOCK_FILTERS[@]}" -gt 0 ]]; then
	echo "[citynav-rasterize] blocks=${BLOCK_FILTERS[*]}"
fi
i=0
for d in "${PLY_DIRS[@]}"; do
	i=$((i + 1))
	source_dir="$d"
	if [[ "${#BLOCK_FILTERS[@]}" -gt 0 ]]; then
		source_dir="$WORKDIR/ply_filter_$i"
		mkdir -p "$source_dir"
		for block in "${BLOCK_FILTERS[@]}"; do
			if [[ -f "$d/$block.ply" ]]; then
				ln -sf "$d/$block.ply" "$source_dir/$block.ply"
			fi
		done
	fi
	n="$(find "$source_dir" -maxdepth 1 -name '*.ply' | wc -l)"
	echo "[citynav-rasterize] rasterizing $n PLYs from $d ..."
	(cd "$WORKDIR" && "${RASTERIZE_CMD[@]}" "$RASTERIZE_PY" "$source_dir" "$TARGET_DIR")
done

N_TIF="$(find "$TARGET_DIR" -maxdepth 1 -name '*.tif' | wc -l)"
N_PNG="$(find "$TARGET_DIR" -maxdepth 1 -name '*.png' | wc -l)"
echo "[citynav-rasterize] done: $N_TIF .tif + $N_PNG .png in $TARGET_DIR"
echo "[citynav-rasterize] the ortho path reads this dir."
