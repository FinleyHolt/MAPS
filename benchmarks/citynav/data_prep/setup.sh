#!/usr/bin/env bash
# CityNav from nothing to a bench that runs, from the official sources alone:
# the annotations (CityNav's own download), the block point clouds
# (SensatUrban's own release), the nadir rasters (CityNav's rasterizer) and
# the tile bake (the OpenStreetMap caches this repository ships). Each step
# is skipped where its output is already on disk.
#
#   benchmarks/citynav/data_prep/setup.sh --sensaturban <data_release.zip | its unpacked folder>
#
# SensatUrban is the one thing no script can fetch: its authors release it
# through a form (https://github.com/QingyongHu/SensatUrban, "Download"),
# whose confirmation page links data_release.zip. Its terms are CC BY-NC-SA,
# non-commercial research use, and no redistribution in any format, so this
# repository ships no point cloud, raster or picture made from one, and
# names no mirror. Run with no argument, the script fetches the annotations
# and says what it is waiting for.
#
# Reads CITYNAV_ROOT (default ~/datasets/citynav-arxiv), PY (the repository's
# python, default .venv/bin/python then python3) and RASTERIZE_PYTHON (an
# interpreter with pdal, rasterio and pillow; without one, a conda-family
# tool on PATH builds the `rasterize` environment upstream CityNav names).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"
export CITYNAV_ROOT="${SEARCH_CITYNAV_DATA:-${CITYNAV_ROOT:-$HOME/datasets/citynav-arxiv}}"
if [[ -z "${PY:-}" ]]; then
	if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY=python3; fi
fi
export PY
say() { printf '[citynav-setup] %s\n' "$*"; }
die() {
	printf '[citynav-setup] ERROR: %s\n' "$*" >&2
	exit 1
}

RELEASE=""
case "${1:-}" in
"") ;;
--sensaturban)
	RELEASE="${2:-}"
	[[ -e "$RELEASE" ]] || die "--sensaturban takes data_release.zip or its unpacked folder; '$RELEASE' is neither"
	;;
*) die "unknown argument '$1' (see this file's header)" ;;
esac

say "1/4 annotations -> $CITYNAV_ROOT"
if [[ -f "$CITYNAV_ROOT/data/citynav/citynav_val_unseen.json" && -f "$CITYNAV_ROOT/data/cityrefer/objects.json" ]]; then
	say "    present"
else
	[[ -d "$CITYNAV_ROOT/.git" ]] || git clone --depth 1 https://github.com/water-cookie/citynav-arxiv "$CITYNAV_ROOT"
	(cd "$CITYNAV_ROOT" && sh scripts/download_data.sh)
	[[ -f "$CITYNAV_ROOT/data/citynav/citynav_val_unseen.json" ]] || die "upstream's download_data.sh left no splits under $CITYNAV_ROOT/data/citynav"
fi

say "2/4 block point clouds -> $CITYNAV_ROOT/raw_ply"
mkdir -p "$CITYNAV_ROOT/raw_ply/train" "$CITYNAV_ROOT/raw_ply/test"
if [[ -n "$RELEASE" ]]; then
	if [[ -f "$RELEASE" ]]; then
		UNPACKED="$CITYNAV_ROOT/raw_ply/data_release"
		[[ -d "$UNPACKED" ]] || (mkdir -p "$UNPACKED" && unzip -q "$RELEASE" -d "$UNPACKED")
		RELEASE="$UNPACKED"
	fi
	# The release holds train/ and test/ somewhere under its top folder.
	for split in train test; do
		while IFS= read -r ply; do
			ln -sf "$(realpath "$ply")" "$CITYNAV_ROOT/raw_ply/$split/$(basename "$ply")"
		done < <(find "$RELEASE" -path "*/$split/*.ply")
	done
fi
N_PLY="$(find "$CITYNAV_ROOT/raw_ply/train" "$CITYNAV_ROOT/raw_ply/test" -maxdepth 1 -name '*.ply' | wc -l)"
if [[ "$N_PLY" -eq 0 ]]; then
	say "    none yet. Request SensatUrban at https://github.com/QingyongHu/SensatUrban (its Download form),"
	say "    then: $0 --sensaturban <data_release.zip>"
	exit 3
fi
say "    $N_PLY block point clouds"

say "3/4 nadir rasters (upstream's PDAL rasterizer)"
RGBD_DIR="$("$PY" -c 'import os; from benchmarks.citynav.paths import citynav_bench; print(citynav_bench(os.environ).rgbd_dir)')"
TODO=()
while IFS= read -r ply; do
	block="$(basename "$ply" .ply)"
	[[ -f "$RGBD_DIR/$block.tif" && -f "$RGBD_DIR/$block.png" ]] || TODO+=("$block")
done < <(find "$CITYNAV_ROOT/raw_ply/train" "$CITYNAV_ROOT/raw_ply/test" -maxdepth 1 -name '*.ply' | sort)
if [[ "${#TODO[@]}" -eq 0 ]]; then
	say "    present"
else
	if [[ -z "${RASTERIZE_PYTHON:-}" ]]; then
		for root in "$HOME/micromamba/envs" "$HOME/.local/share/mamba/envs" "$HOME/miniconda3/envs" "$HOME/miniforge3/envs" "$HOME/anaconda3/envs"; do
			if [[ -x "$root/rasterize/bin/python" ]]; then
				RASTERIZE_PYTHON="$root/rasterize/bin/python"
				break
			fi
		done
	fi
	if [[ -z "${RASTERIZE_PYTHON:-}" ]]; then
		TOOL=""
		for t in micromamba mamba conda; do
			if command -v "$t" >/dev/null 2>&1; then
				TOOL="$t"
				break
			fi
		done
		[[ -n "$TOOL" ]] || die "the rasterizer needs PDAL: install micromamba (or conda) and rerun, or set RASTERIZE_PYTHON to an interpreter with pdal, rasterio and pillow"
		say "    building the rasterize environment with $TOOL"
		"$TOOL" create --yes --name rasterize --channel conda-forge pdal python=3.10 rasterio pillow
		RASTERIZE_PYTHON="$("$TOOL" run --name rasterize python -c 'import sys; print(sys.executable)')"
	fi
	export RASTERIZE_PYTHON
	say "    ${#TODO[@]} blocks to rasterize: ${TODO[*]}"
	"$HERE/rasterize_citynav_native.sh" --blocks "${TODO[@]}"
fi

say "4/4 tile bake"
"$HERE/bake_tiles.sh"

"$PY" - <<'PY'
import collections
import os

from benchmarks.citynav.adapter import load_split, parse_episode
from benchmarks.citynav.paths import citynav_bench

bench = citynav_bench(os.environ)
baked = {p.name.removesuffix("_meta.json") for p in bench.tiles_dir.glob("*_meta.json")}
print(f"[citynav-setup] {len(baked)} blocks baked under {bench.tiles_dir}; bake_sha256 {bench.bake_sha256()}")
for split in ("val_unseen", "test_unseen", "val_seen", "train_seen"):
    blocks = collections.Counter(parse_episode(r, row_index=i).map_name for i, r in enumerate(load_split(bench.split_path(split))))
    held = sum(n for b, n in blocks.items() if b in baked)
    missing = sorted(b for b in blocks if b not in baked)
    word = "ready" if not missing else f"lacks {len(missing)} of its {len(blocks)} blocks"
    print(f"[citynav-setup]   {split}: {held}/{sum(blocks.values())} episodes on baked blocks, {word}")
print("[citynav-setup] fly a split that reads ready:")
print("[citynav-setup]   python -m benchmarks.citynav.bench_fullset --split test_unseen --limit 100 --out-root output/citynav")
PY
