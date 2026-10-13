#!/usr/bin/env bash
# One stage of a released CityNav baseline. README.md beside this script is
# the procedure; the stages:
#
#   setup [all|mgp|hett|flightgpt]       clones, envs, checkpoints, data
#   mgp-citynav                          val_seen, val_unseen, test_unseen
#   hett-citynav original|revised        val_seen, val_unseen, test_unseen
#   flightgpt-citynav                    test_unseen + val_unseen tiers (4 GPUs)
#   calcity-export                       per-episode blocks (CPU)
#   calcity-parse                        processed_descriptions.json (needs RELEASED_PARSE_URL)
#   mgp-calcity | hett-calcity | flightgpt-calcity   every export shard
#
# Each run stage converts its outputs to rows under
# $RELEASED_OUT/<method>_<bench>_<split>[_<variant>]/.
#
# Environment:
#   RELEASED_ROOT  (required) clones, envs, checkpoints, data and upstream run
#                  roots; setup needs about 150 GB free there
#   RELEASED_OUT   the rows (default output/released_baselines under the repo)
#   BENCH_PY       this repo's Python (default: python on PATH)
#   MICROMAMBA     the micromamba binary the method envs are built with
#
# Run it on a GPU node, never a shared login host: setup builds three CUDA
# environments and downloads about 150 GB, and every run stage loads a
# checkpoint. Under slurm, wrap it: sbatch --gres=gpu:1 --wrap "$0 mgp-citynav".
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
: "${RELEASED_ROOT:?set RELEASED_ROOT to the directory the released baselines live in}"
OUT="${RELEASED_OUT:-$REPO_DIR/output/released_baselines}"
PY="${BENCH_PY:-python}"
MM="${MICROMAMBA:-micromamba}"
EXPORT="$RELEASED_ROOT/calcity/export"
LAUNCH="$REPO_DIR/baselines/released/upstream_eval/launch.py"
envdir() { echo "$RELEASED_ROOT/envs/released-$1"; }
envpy() { echo "$(envdir "$1")/bin/python"; }

# Upstream download links, as each README publishes them (dl=1 asks Dropbox
# for the file, or for a folder as one zip).
MGP_CKPT_URL='https://www.dropbox.com/scl/fi/xoz7mzv4fddwktn961chg/mgp_mturk.pth?rlkey=r80r51gtu38mfx022md3gmq7m&dl=1'
HETT_DROPBOX_URL='https://www.dropbox.com/scl/fo/ua3kn6mw3adn2hcsdikt3/AEdoRoo-OeXnbrpeOpk4BcQ?rlkey=v5rlqiqa1k9ml8isinpv7glgk&st=ahseq0r2&dl=1'

say() { echo "== [released] $*"; }
die() { echo "released: $*" >&2; exit 1; }

stage="${1:?stage; see the header}"
shift
mkdir -p "$RELEASED_ROOT/src" "$OUT"
echo "== released $stage $(date -u +%FT%TZ) node=$(hostname) commit=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')"

# ---- setup ------------------------------------------------------------------

pin() { "$PY" -m baselines.released.upstream pins | awk -v m="$1" -v f="$2" '$1==m {print $f}'; }

clone() { # method
	local dir="$RELEASED_ROOT/src/$1" sha
	sha="$(pin "$1" 3)"
	[ -d "$dir/.git" ] || git clone "$(pin "$1" 2)" "$dir"
	git -C "$dir" fetch -q origin "$sha" 2>/dev/null || true
	git -C "$dir" checkout -q "$sha"
	[ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || die "$1 clone is not at $sha"
}

mkenv() { # method python
	local env
	env="$(envdir "$1")"
	[ -x "$env/bin/python" ] || "$MM" create -y -p "$env" -c conda-forge "python=$2" pip
	echo "$env"
}

sha_id() { echo "$(basename "$1") sha256:$(sha256sum "$1" | cut -d' ' -f1)"; }

link_rgbd() { # dest_dir: this repo's SensatUrban rasters, else FlightGPT's rgbd-new
	local src f
	src="$("$PY" -c 'from benchmarks.citynav import paths; print(paths.CITYNAV_NATIVE_RGBD_DIR)')"
	[ -d "$src" ] || src="$RELEASED_ROOT/flightgpt/data_hf/rgbd-new"
	[ -d "$src" ] || die "no SensatUrban rasters: neither the bench's rgbd_upstream nor flightgpt rgbd-new"
	mkdir -p "$1"
	for f in "$src"/*.png "$src"/*.tif; do ln -sfn "$f" "$1/"; done
}

setup_mgp() {
	clone mgp
	local src="$RELEASED_ROOT/src/mgp" root="$RELEASED_ROOT/mgp" env
	env="$(mkenv mgp 3.10)"
	if [ ! -f "$env/.released_ok" ]; then
		say "mgp env (CityNav README order)"
		"$MM" install -y -p "$env" -c pytorch -c nvidia -c conda-forge \
			pytorch=2.2.2 torchvision=0.17.2 pytorch-cuda=11.8 mpi4py
		local pip="$env/bin/pip"
		"$pip" install git+https://github.com/water-cookie/Segment-Everything-Everywhere-All-At-Once.git@package
		"$pip" install git+https://github.com/water-cookie/Semantic-SAM.git@package
		"$pip" install git+https://github.com/facebookresearch/segment-anything.git
		[ -d "$RELEASED_ROOT/src/SoM" ] || git clone https://github.com/water-cookie/SoM.git "$RELEASED_ROOT/src/SoM"
		(cd "$RELEASED_ROOT/src/SoM/ops" && PATH="$env/bin:$PATH" ./make.sh)
		"$pip" install --editable "$RELEASED_ROOT/src/SoM"
		"$pip" install git+https://github.com/water-cookie/LLaVA.git
		"$pip" install git+https://github.com/IDEA-Research/GroundingDINO.git
		"$pip" install git+https://github.com/ChaoningZhang/MobileSAM.git
		"$pip" install -r "$src/requirements.txt"
		touch "$env/.released_ok"
	fi
	mkdir -p "$root"
	# Their scripts write data/ and weights/ relative to the working directory.
	[ -f "$root/data/cityrefer/processed_descriptions.json" ] || (cd "$root" && sh "$src/scripts/download_data.sh")
	[ -f "$root/weights/mobile_sam/mobile_sam.pt" ] || (cd "$root" && PATH="$env/bin:$PATH" sh "$src/scripts/download_weights.sh") || true
	for w in groundingdino/groundingdino_swint_ogc.pth mobile_sam/mobile_sam.pt \
		vlnce/data/ddppo-models/gibson-2plus-resnet50.pth; do
		[ -f "$root/weights/$w" ] || die "mgp weight missing: weights/$w"
	done
	local ckpt="$root/checkpoints/mgp_mturk/mgp_mturk.pth"
	[ -f "$ckpt" ] || { mkdir -p "$(dirname "$ckpt")"; wget -q -O "$ckpt" "$MGP_CKPT_URL"; }
	sha_id "$ckpt" >"$root/checkpoint.id"
	link_rgbd "$root/data/rgbd"
	local run="$root/run_citynav"
	mkdir -p "$run"
	ln -sfn "$root/data" "$run/data"
	ln -sfn "$root/weights" "$run/weights"
	ln -sfn "$src/gsamllavanav" "$run/gsamllavanav"
	say "mgp ready: $(cat "$root/checkpoint.id")"
}

setup_hett() {
	clone hett
	[ -d "$RELEASED_ROOT/mgp/data/cityrefer" ] || die "hett reads CityNav's data download; run setup mgp first"
	local src="$RELEASED_ROOT/src/hett" root="$RELEASED_ROOT/hett" env
	env="$(mkenv hett 3.10)"
	if [ ! -f "$env/.released_ok" ]; then
		say "hett env"
		"$MM" install -y -p "$env" -c pytorch -c nvidia -c conda-forge \
			pytorch=2.2.2 torchvision=0.17.2 pytorch-cuda=11.8
		# requirements.txt is a frozen conda env: drop what pip cannot build
		# or what the conda install already pinned.
		mkdir -p "$root"
		grep -viE '^(apex|detectron2|MultiScaleDeformableAttention|mkl[-_]|task-adapter|typing==|gmpy2|deepspeed|torch|torchvision|triton|pip|setuptools|wheel)\b' \
			"$src/requirements.txt" >"$root.requirements.txt"
		"$env/bin/pip" install -r "$root.requirements.txt"
		"$env/bin/python" -c \
			"from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased'); AutoModel.from_pretrained('bert-base-uncased')"
		touch "$env/.released_ok"
	fi
	mkdir -p "$root/dropbox"
	if [ ! -f "$root/dropbox/.unzipped" ]; then
		wget -q -O "$root/dropbox.zip" "$HETT_DROPBOX_URL"
		unzip -q -o "$root/dropbox.zip" -d "$root/dropbox" && touch "$root/dropbox/.unzipped"
	fi
	# Nested archives inside the folder zip unpack beside themselves.
	find "$root/dropbox" -name '*.zip' -print0 | while IFS= read -r -d '' z; do
		[ -e "${z%.zip}.unzipped" ] || { unzip -q -o "$z" -d "$(dirname "$z")" && touch "${z%.zip}.unzipped"; }
	done
	local ckpt cfg yolo revised
	ckpt="$(find "$root/dropbox" -name best_val_unseen -type f | head -1)"
	cfg="$(find "$root/dropbox" -name yolo_v3.cfg | head -1)"
	yolo="$(find "$root/dropbox" -name best.pt | head -1)"
	revised="$(dirname "$(find "$root/dropbox" -name citynav_test_unseen.json | head -1)")"
	[ -n "$ckpt" ] || die "no best_val_unseen in the HETT Dropbox folder (see $root/dropbox)"
	[ -n "$cfg" ] && [ -n "$yolo" ] || die "no yolo_v3.cfg / best.pt in the HETT Dropbox folder"
	[ "$revised" != . ] || die "no refined citynav split in the HETT Dropbox folder"
	sha_id "$ckpt" >"$root/checkpoint.id"
	ln -sfn "$ckpt" "$root/best_val_unseen"
	ln -sfn "$revised" "$root/refined_citynav"
	for variant in original revised; do
		local run="$root/run_$variant"
		mkdir -p "$run/work" "$run/data" "$run/weights" "$run/checkpoints/multi"
		if [ "$variant" = original ]; then
			ln -sfn "$RELEASED_ROOT/mgp/data/citynav" "$run/data/processed_citynav"
		else
			ln -sfn "$revised" "$run/data/processed_citynav"
		fi
		ln -sfn "$RELEASED_ROOT/mgp/data/cityrefer" "$run/data/cityrefer"
		ln -sfn "$RELEASED_ROOT/mgp/data/rgbd" "$run/data/rgbd"
		ln -sfn "$cfg" "$run/weights/yolo_v3.cfg"
		ln -sfn "$yolo" "$run/weights/best.pt"
	done
	say "hett ready: $(cat "$root/checkpoint.id")"
}

setup_flightgpt() {
	clone flightgpt
	[ -d "$RELEASED_ROOT/mgp/data/cityrefer" ] || die "flightgpt reads CityNav's cityrefer; run setup mgp first"
	local src="$RELEASED_ROOT/src/flightgpt" root="$RELEASED_ROOT/flightgpt" env venv
	env="$(mkenv flightgpt 3.11)"
	if [ ! -f "$env/.released_ok" ]; then
		say "flightgpt env (FlightGPT README)"
		"$env/bin/pip" install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
			--index-url https://download.pytorch.org/whl/cu124
		"$env/bin/pip" install -r "$src/requirements.txt" huggingface_hub
		# gsamllavanav/maps/gsam_map.py imports groundingdino at module level.
		"$env/bin/pip" install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git
		touch "$env/.released_ok"
	fi
	venv="$(mkenv flightgpt-vllm 3.11)"
	[ -x "$venv/bin/vllm" ] || "$venv/bin/pip" install "vllm==0.8.5"
	local rev
	rev="$("$env/bin/python" -c "from huggingface_hub import HfApi; print(HfApi().model_info('ADJHD/Flightgpt').sha)")"
	"$env/bin/hf" download ADJHD/Flightgpt --revision "$rev" --include 'Flightgpt/*' \
		--local-dir "$root/model_weight" >/dev/null
	"$env/bin/python" - "$root/model_weight/Flightgpt/preprocessor_config.json" <<-'PY'
		import json, sys
		cfg = json.load(open(sys.argv[1]))
		assert cfg["max_pixels"] == 16032016, f"max_pixels {cfg['max_pixels']}, README wants 16032016"
	PY
	echo "ADJHD/Flightgpt@$rev Flightgpt/" >"$root/checkpoint.id"
	local drev
	drev="$("$env/bin/python" -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('ADJHD/flightgpt_training_data').sha)")"
	"$env/bin/hf" download ADJHD/flightgpt_training_data --repo-type dataset --revision "$drev" \
		--include 'citynav/*' 'gsam/*' 'rgbd-new/*' --local-dir "$root/data_hf" >/dev/null
	echo "$drev" >"$root/data_hf/revision"
	local run="$root/run_citynav"
	mkdir -p "$run/data"
	ln -sfn "$root/data_hf/citynav" "$run/data/citynav"
	ln -sfn "$root/data_hf/gsam" "$run/data/gsam"
	ln -sfn "$root/data_hf/rgbd-new" "$run/data/rgbd-new"
	ln -sfn "$RELEASED_ROOT/mgp/data/cityrefer" "$run/data/cityrefer"
	say "flightgpt ready: $(cat "$root/checkpoint.id")"
}

# ---- run stages ---------------------------------------------------------------

convert_citynav() { # method split native... -- loaded...   (env: VERSION, ID_FILE, TAG)
	local method="$1" split="$2"
	shift 2
	local natives=() loaded=()
	while [ $# -gt 0 ] && [ "$1" != -- ]; do natives+=("$1"); shift; done
	[ "${1:-}" = -- ] && shift
	[ $# -gt 0 ] && loaded=(--loaded "$@")
	"$PY" -m baselines.released.outcome_rows citynav --method "$method" \
		--native "${natives[@]}" --split "$split" --split-file "$ID_FILE" \
		"${loaded[@]}" --dataset-version "$VERSION" \
		--checkpoint-id "$(cat "$RELEASED_ROOT/$method/checkpoint.id")" \
		--out "$OUT/${method}_citynav_${split}${TAG:-}"
}

convert_calcity() { # method native...
	local method="$1"
	shift
	"$PY" -m baselines.released.outcome_rows calcity --method "$method" \
		--native "$@" --export "$EXPORT" \
		--checkpoint-id "$(cat "$RELEASED_ROOT/$method/checkpoint.id")" \
		--out "$OUT/${method}_calcity"
}

# A Cal City shard as an upstream run root: data/ from the export, weights
# and code configs from the method's CityNav root.
shard_root() { # method shard_dir -> echoes the run root
	local method="$1" shard="$2" run
	run="$RELEASED_ROOT/calcity/$method/$(basename "$shard")"
	mkdir -p "$run"
	case "$method" in
	mgp)
		ln -sfn "$shard/data" "$run/data"
		ln -sfn "$RELEASED_ROOT/mgp/weights" "$run/weights"
		ln -sfn "$RELEASED_ROOT/src/mgp/gsamllavanav" "$run/gsamllavanav"
		;;
	hett)
		mkdir -p "$run/work" "$run/data" "$run/checkpoints/multi"
		ln -sfn "$shard/data/citynav" "$run/data/processed_citynav"
		ln -sfn "$shard/data/cityrefer" "$run/data/cityrefer"
		ln -sfn "$shard/data/rgbd" "$run/data/rgbd"
		ln -sfn "$RELEASED_ROOT/hett/run_original/weights" "$run/weights"
		;;
	flightgpt)
		mkdir -p "$run/data"
		ln -sfn "$shard/data/citynav" "$run/data/citynav"
		ln -sfn "$shard/data/cityrefer" "$run/data/cityrefer"
		ln -sfn "$shard/data/rgbd" "$run/data/rgbd-new"
		;;
	esac
	echo "$run"
}

hett_flags() { # the flags multiagent/eval.sh passes, at world_size 1
	echo --world_size 1 --seed 0 --feedback student --mode eval --altitude 50 \
		--learning_rate 1e-4 --batch_size 2 --train_trajectory_type mturk \
		--log_every 1 --eval_every 1 --epochs 50 --save_every 1 --log_dir log \
		--move_iteration 10 --max_action_len 20 --grid_size 5 \
		--checkpoint "$RELEASED_ROOT/hett/best_val_unseen"
}

# FlightGPT's vLLM, exactly the README's serve command, for the length of
# this stage. Started by the stage, stopped by the trap.
VLLM_PID=""
stop_vllm() { [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null || true; }
trap stop_vllm EXIT
serve_flightgpt() {
	local port=8888 tp
	tp="$(nvidia-smi -L | wc -l)"
	[ "$tp" -ge 4 ] || echo "   WARNING: $tp GPUs; FlightGPT's README serves at -tp 4" >&2
	"$(envdir flightgpt-vllm)/bin/vllm" serve \
		"$RELEASED_ROOT/flightgpt/model_weight/Flightgpt" \
		--dtype auto --trust-remote-code --served-model-name qwen_2_5_vl_7b \
		--host 127.0.0.1 --port "$port" -tp "$tp" \
		--limit-mm-per-prompt image=2,video=0 --max-model-len=32000 &
	VLLM_PID=$!
	export FLIGHTGPT_API_BASE="http://127.0.0.1:$port/v1" FLIGHTGPT_MODEL=qwen_2_5_vl_7b
	for _ in $(seq 180); do
		curl -sf "$FLIGHTGPT_API_BASE/models" >/dev/null && return 0
		kill -0 "$VLLM_PID" 2>/dev/null || die "vLLM exited"
		sleep 10
	done
	die "vLLM not up after 30 min"
}

case "$stage" in
setup)
	avail_gb=$(df -BG --output=avail "$RELEASED_ROOT" | tail -1 | tr -dc 0-9)
	[ "$avail_gb" -ge 150 ] || die "$RELEASED_ROOT has ${avail_gb} GB free; the three methods need about 150 GB"
	case "${1:-all}" in
	all) setup_mgp; setup_hett; setup_flightgpt ;;
	mgp) setup_mgp ;;
	hett) setup_hett ;;
	flightgpt) setup_flightgpt ;;
	*) die "usage: $0 setup all|mgp|hett|flightgpt" ;;
	esac
	;;

mgp-citynav)
	run="$RELEASED_ROOT/mgp/run_citynav"
	(cd "$run" && "$(envpy mgp)" "$RELEASED_ROOT/src/mgp/main_goal_predictor.py" \
		--mode eval --altitude 50 --gsam_use_segmentation_mask --gsam_box_threshold 0.20 \
		--eval_batch_size 200 --eval_max_timestep 20 \
		--checkpoint "$RELEASED_ROOT/mgp/checkpoints/mgp_mturk/mgp_mturk.pth")
	for split in val_seen val_unseen test_unseen; do
		ID_FILE="$run/data/citynav/citynav_$split.json" VERSION=original \
			convert_citynav mgp "$split" "$run/mgp_mgp_mturk_${split}_0.75_gdino.json"
	done
	;;

hett-citynav)
	variant="${1:?original|revised}"
	run="$RELEASED_ROOT/hett/run_$variant"
	# shellcheck disable=SC2046
	(cd "$run/work" && HETT_BERT=bert-base-uncased "$(envpy hett)" "$LAUNCH" --method hett \
		--clone "$RELEASED_ROOT/src/hett" --capture-dir "$run/captures" -- $(hett_flags))
	for split in val_seen val_unseen test_unseen; do
		ID_FILE="$run/data/processed_citynav/citynav_$split.json" VERSION="$variant" TAG="_$variant" \
			convert_citynav hett "$split" "$run/captures/hett_$split.json"
	done
	;;

flightgpt-citynav)
	run="$RELEASED_ROOT/flightgpt/run_citynav"
	serve_flightgpt
	cn="$run/data/citynav"
	(cd "$run" && "$(envpy flightgpt)" "$LAUNCH" --method flightgpt \
		--clone "$RELEASED_ROOT/src/flightgpt" --capture-dir "$run/captures" \
		--split-files "$cn/citynav_test_unseen.json" "$cn/citynav_val_unseen_easy.json" \
		"$cn/citynav_val_unseen_medium.json" "$cn/citynav_val_unseen_hard.json")
	ID_FILE="$cn/citynav_test_unseen.json" VERSION=original \
		convert_citynav flightgpt test_unseen "$run/captures/flightgpt_test_unseen.json"
	ID_FILE="$cn/citynav_val_unseen.json" VERSION=original convert_citynav flightgpt val_unseen \
		"$run/captures"/flightgpt_val_unseen_{easy,medium,hard}.json -- \
		"$cn"/citynav_val_unseen_{easy,medium,hard}.json
	;;

calcity-export)
	"$PY" -m baselines.released.calcity_export --out "$EXPORT" "$@"
	;;

calcity-parse)
	: "${RELEASED_PARSE_URL:?point RELEASED_PARSE_URL at an OpenAI-compatible /v1 endpoint}"
	"$PY" -m baselines.released.describe_instructions --export "$EXPORT"
	;;

mgp-calcity | hett-calcity | flightgpt-calcity)
	method="${stage%-calcity}"
	[ -f "$EXPORT/processed_descriptions.json" ] || die "run calcity-parse first"
	[ "$method" = flightgpt ] && serve_flightgpt
	captures=()
	for shard in "$EXPORT"/shards/shard_*; do
		run="$(shard_root "$method" "$shard")"
		case "$method" in
		mgp)
			(cd "$run" && "$(envpy mgp)" "$LAUNCH" --method mgp --clone "$RELEASED_ROOT/src/mgp" \
				--blocks "$EXPORT/blocks.json" --splits test_unseen -- \
				--mode eval --altitude 50 --gsam_use_segmentation_mask --gsam_box_threshold 0.20 \
				--eval_batch_size 200 --eval_max_timestep 20 \
				--checkpoint "$RELEASED_ROOT/mgp/checkpoints/mgp_mturk/mgp_mturk.pth")
			captures+=("$run/mgp_mgp_mturk_test_unseen_0.75_gdino.json")
			;;
		hett)
			# shellcheck disable=SC2046
			(cd "$run/work" && HETT_BERT=bert-base-uncased "$(envpy hett)" "$LAUNCH" --method hett \
				--clone "$RELEASED_ROOT/src/hett" --blocks "$EXPORT/blocks.json" \
				--splits test_unseen --capture-dir "$run/captures" -- $(hett_flags))
			captures+=("$run/captures/hett_test_unseen.json")
			;;
		flightgpt)
			(cd "$run" && "$(envpy flightgpt)" "$LAUNCH" --method flightgpt \
				--clone "$RELEASED_ROOT/src/flightgpt" --blocks "$EXPORT/blocks.json" \
				--zero-gsam-cache --capture-dir "$run/captures" \
				--split-files "$run/data/citynav/citynav_test_unseen.json")
			captures+=("$run/captures/flightgpt_test_unseen.json")
			;;
		esac
	done
	convert_calcity "$method" "${captures[@]}"
	;;

*)
	sed -n '2,25p' "$0" >&2
	exit 2
	;;
esac
echo "== exit 0 ($(date -u +%FT%TZ))"
