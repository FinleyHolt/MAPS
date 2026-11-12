#!/usr/bin/env bash
# Fetch the CityNav benchmark data into $CITYNAV_ROOT. Nothing is
# redistributed by this repo: everything comes from the canonical
# upstream sources, then benchmarks/citynav/data_prep/rasterize_citynav_native.sh
# and bake_tiles.sh regenerate the rasters and tile sidecars locally.
#
# Layout produced (what benchmarks/citynav/paths.py expects):
#   $CITYNAV_ROOT/data/citynav/citynav_{train_seen,val_*,test_unseen}.json
#   $CITYNAV_ROOT/data/cityrefer/objects.json
#   $CITYNAV_ROOT/raw_ply/{train,test}/<block>.ply   (SensatUrban)
#
# Usage:
#   CITYNAV_ROOT=~/datasets/citynav-arxiv benchmarks/citynav/data_prep/fetch_data.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CITYNAV_ROOT="${SEARCH_CITYNAV_DATA:-${CITYNAV_ROOT:-$HOME/datasets/citynav-arxiv}}"

echo "== 1/4: upstream CityNav checkout -> ${CITYNAV_ROOT}"
if [[ -d "${CITYNAV_ROOT}/.git" ]]; then
	echo "   already cloned; reusing."
else
	git clone https://github.com/water-cookie/citynav-arxiv "${CITYNAV_ROOT}"
fi

echo "== 2/4: episode splits + CityRefer objects.json (upstream downloader)"
(cd "${CITYNAV_ROOT}" && sh scripts/download_data.sh) ||
	echo "   WARNING: upstream download_data.sh failed; re-run it manually inside ${CITYNAV_ROOT}"

echo "== 3/4: SensatUrban block PLYs (manual: license-gated upstream)"
PLY_COUNT="$(find "${CITYNAV_ROOT}/raw_ply" -name '*.ply' 2>/dev/null | wc -l)"
echo "   present: ${PLY_COUNT}/34 block PLYs under ${CITYNAV_ROOT}/raw_ply/"
if [[ "${PLY_COUNT}" -lt 34 ]]; then
	cat <<'EOF'
   The 34 SensatUrban blocks are NOT auto-fetched: their authors release
   them through a form (https://github.com/QingyongHu/SensatUrban, "Download"),
   under CC BY-NC-SA, for non-commercial research, with no redistribution.
   benchmarks/citynav/data_prep/setup.sh --sensaturban <data_release.zip>
   unpacks, rasterizes and bakes them in one step.
   Place them at $CITYNAV_ROOT/raw_ply/{train,test}/<block>.ply.
   The tile bake reads its blocks off the rasters made from them: run
   benchmarks/citynav/data_prep/rasterize_citynav_native.sh, then
   benchmarks/citynav/data_prep/bake_tiles.sh.
EOF
fi

echo "== 4/4: model weights (fetched by their own tooling)"
cat <<'EOF'
   - SAM 3.1 (gated): approve https://huggingface.co/facebook/sam3.1 then
     export HF_TOKEN (the SAM serve fetches the checkpoint on first load).
   - Reasoner: any OpenAI-compatible endpoint (SEARCH_REASONER_URL).
EOF

echo
echo "== verify =="
if python3 -c "import os; from benchmarks.citynav.paths import citynav_bench; citynav_bench(os.environ).assert_available()" 2>/dev/null; then
	echo "paths.assert_available(): OK: splits + objects.json present."
else
	echo "paths.assert_available(): NOT YET (see messages above; splits/objects missing)."
	exit 1
fi
