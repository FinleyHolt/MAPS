# Released CityNav baselines

Three published checkpoints run under their authors' own evaluators, on
CityNav under each method's released protocol and on the Cal City bench.
Every outcome lands as ledger rows that the canonical readers in
`benchmarks/citynav/analysis/` accept, keyed by the harness's own episode
ids, so a checkpoint pairs against a MAPS run on the same episodes.

| Method | Upstream | Pinned commit | Checkpoint |
|---|---|---|---|
| CityNav MGP | `water-cookie/citynav` | `372ecbd1` | `mgp_mturk.pth` (MGP w/ HD) |
| HETT | `crotonyl/HETT` (`multiagent/`) | `51eb86c3` | `checkpoints/multi/best_val_unseen`, Dropbox folder |
| FlightGPT | `Uavln/FlightGPT` | `a1dd6760` | `ADJHD/Flightgpt`, subfolder `Flightgpt/` |

The full commit shas live in `upstream.py`, the one table the setup script
clones from and every row is stamped with. The checkpoint identity on a row
is measured at setup (`<method>/checkpoint.id`: a sha256 of the weights file,
or the Hugging Face revision), never written down by hand.

## What runs where

| Piece | Runs under | Purpose |
|---|---|---|
| `released.sh` | a GPU node | one stage per call: setup, a run, an export, a parse |
| `upstream_eval/launch.py` | each method's own env (Python 3.10/3.11) | run-time adaptation of the untouched clone |
| `calcity_export.py` | this repo's env, CPU | Cal City as per-episode `gsamllavanav` blocks |
| `describe_instructions.py` | this repo's env, CPU + an LLM endpoint | `processed_descriptions.json` for Cal City |
| `outcome_rows.py` | this repo's env, CPU | native outputs to `episodes.jsonl` + `episode_sets.json`, and the pair against a MAPS run |

The upstream clones are never edited. Each adaptation `launch.py` makes is
listed in its docstring and in the deviation tables below.

## Rerun

`released.sh` reads `RELEASED_ROOT` (clones, `micromamba` envs, checkpoints,
data and upstream run roots: about 150 GB) and writes rows under
`output/released_baselines/` (`RELEASED_OUT`). Run every stage on a GPU
node; under slurm, wrap each one (`sbatch --gres=gpu:1 --wrap
"baselines/released/released.sh mgp-citynav"`).

```sh
export RELEASED_ROOT=/data/released_baselines
R=baselines/released/released.sh
$R setup mgp                 # first: HETT and FlightGPT reuse CityNav's data download
$R setup hett
$R setup flightgpt

$R mgp-citynav               # val_seen, val_unseen, test_unseen
$R hett-citynav original
$R hett-citynav revised
$R flightgpt-citynav         # test_unseen + val_unseen easy/medium/hard, on 4 GPUs

$R calcity-export
RELEASED_PARSE_URL=http://<host>:<port>/v1 $R calcity-parse
$R mgp-calcity
$R hett-calcity
$R flightgpt-calcity         # on 4 GPUs
```

Read an outcome with the bench's reader, and pair it against a MAPS run on
the same split file:

```sh
python -m benchmarks.citynav.analysis.run_report output/released_baselines/mgp_citynav_test_unseen
python -m baselines.released.outcome_rows pair output/released_baselines/mgp_citynav_test_unseen <maps_run>
```

The MAPS run beside a released checkpoint on CityNav flies
`configs/baselines/released_protocol.yaml`, the released evaluators' 50 m
start and 100-action budget. `ab_report` refuses a released row against a
MAPS row: the rows record no tile bake (`search_env_flags` is empty),
because a released checkpoint reads its own rasters, and `ab_report` treats
an unrecorded bake as fatal. `outcome_rows pair` pairs on episode id and
success criterion and prints the same statistics.

## Outputs

| Stage | Native output | Rows |
|---|---|---|
| `mgp-citynav` | `mgp/run_citynav/mgp_mgp_mturk_<split>_0.75_gdino.json` | `mgp_citynav_<split>/` |
| `hett-citynav <v>` | `hett/run_<v>/checkpoints/multi/valid.txt`, `hett/run_<v>/captures/hett_<split>.json` | `hett_citynav_<split>_<v>/` |
| `flightgpt-citynav` | stdout metrics, `flightgpt/run_citynav/captures/flightgpt_<split>.json` | `flightgpt_citynav_{test_unseen,val_unseen}/` |
| `calcity-export` | `calcity/export/{manifest,blocks}.json`, `shards/shard_<k>/data/` | none |
| `<method>-calcity` | the same files per shard under `calcity/<method>/shard_<k>/` | `<method>_calcity/` |

Every row carries `method`, `upstream_repo`, `upstream_sha`, `checkpoint`,
`protocol: upstream_released`, `dataset_version` (`original`, `revised`, or
`calcity`) at the top level and in `provenance`, plus `upstream_key` and
`metrics.success_criterion`: `ring` on CityNav, `shape` on Cal City.

`episode_sets.json` lists the ids upstream evaluated and every id it did
not, with the reason (`upstream_loader_filter`, `upstream_exception`), and on
Cal City the ids the export excluded (`start_and_target_exceed_block`,
`window_leaves_imagery`). `episodes.jsonl` holds the evaluated set only, which
is the set upstream's own metrics are computed on; to score the unfiltered
set, count every listed id as a miss.

## Scoring

`outcome_rows` scores the final pose of each upstream trajectory with
`benchmarks.citynav.eval.score_arrival`:

- CityNav: the ring, 20 m, against the harness's target point (the mean of
  `target_positions`; every released row carries one position, so this is
  upstream's `target_positions[-1]`). Upstream's own number for the same run
  stays in its native output.
- Cal City: the shape criterion `benchmarks.calcity.use` selects, against
  the shape the bench scores the row on (`episode_runner.success_shape`,
  through `calcity_export.calcity_shapes`), exactly as drawn, on the final
  pose (the released methods record no commit). This is the Cal City
  contract, not upstream's 20 m disc.

## CityNav: held to upstream, and ours

### MGP

| Item | Upstream | Ours |
|---|---|---|
| Evaluator | `main_goal_predictor.py --mode eval` | the same file, unmodified, run directly |
| Flags | README eval command: `--altitude 50 --gsam_use_segmentation_mask --gsam_box_threshold 0.20 --eval_batch_size 200 --eval_max_timestep 20` | the same; `--move_iteration 5`, `--progress_stop_val 0.75`, goal selector `gdino` are the parser defaults |
| Splits | `val_seen`, `val_unseen`, `test_unseen` (hard-coded) | the same |
| Data | `scripts/download_data.sh` | the same |
| Rasters | `scripts/rasterize.sh` over SensatUrban | this repo's `rgbd_upstream`, produced by upstream's `rasterize.py` (`benchmarks/citynav/data_prep/rasterize_citynav_native.sh`); FlightGPT's `rgbd-new` when that directory is absent |
| Env | README install order, PyTorch 2.2.2, CUDA 11.8 | the same, at each dependency's current default branch (the README pins none) |

### HETT

| Item | Upstream | Ours |
|---|---|---|
| Evaluator | `multiagent/main.py` via `eval.sh` | `main.main()` through `launch.py` |
| Flags | `eval.sh`: `--max_action_len 20 --move_iteration 10 --altitude 50 --grid_size 5 --batch_size 2 ...` | the same flags, `--world_size 1` |
| Launcher | `torch.distributed.run --nproc_per_node 1` | plain `python`; `main()` skips distributed init at `world_size 1`, so the eval is the same |
| BERT | hard-coded `/cver/xcding/code/tokenizer_files/bert-base-uncase` | `from_pretrained` remapped to `$HETT_BERT` (`bert-base-uncased`) |
| `direction` package | imported by `multiagent/dataset/episode.py`, absent from the repo | aliased to `multiagent` (same modules) |
| Per-episode output | none; `valid.txt` holds averages | trajectories captured at `eval_metrics` |
| Splits | original via `data/processed_citynav`; refined from the Dropbox folder | both, as two run roots (`run_original`, `run_revised`) |
| Env | frozen `requirements.txt`, CUDA 11.3 | PyTorch 2.2.2 on CUDA 11.8 (no 11.3 build of 2.2.2); `requirements.txt` minus `apex`, `detectron2`, `MultiScaleDeformableAttention`, `mkl-*`, `task-adapter`, `typing`, `gmpy2`, `deepspeed` and the conda-pinned torch packages |

### FlightGPT

| Item | Upstream | Ours |
|---|---|---|
| Evaluator | `eval.py` `main` | `eval.run_nav_gym` and `eval_planning_metrics`, driven by `launch.py` |
| Budget | `step_num=2`, `action_num=75` | the same |
| Splits | `val_unseen` easy/medium/hard only | those, plus `test_unseen` (the dataset's own file) |
| Endpoint | hard-coded `http://0.0.0.0:8888/v1` | `$FLIGHTGPT_API_BASE`, the job's own vLLM on `127.0.0.1:8888` |
| Serving | README `vllm serve ... -tp 4 --max-model-len=32000 --limit-mm-per-prompt image=2,video=0` | the same; `-tp` is the job's GPU count, 4 when submitted with `--gres=gpu:4` |
| Decoding | `temperature 0.0`, `max_tokens 2000` (`navgym/agents/CityNavAgent.py`) | unchanged |
| Weights | `ADJHD/Flightgpt`, `max_pixels` 16032016 | the same; setup asserts the shipped `preprocessor_config.json` already says 16032016 |
| Data | `ADJHD/flightgpt_training_data` | the same (`citynav/`, `gsam/`, `rgbd-new/`); `cityrefer/` from CityNav's data download, which the dataset does not ship |
| Env | Python 3.11, PyTorch 2.6.0 cu124, `requirements.txt` | the same, plus GroundingDINO (`gsam_map.py` imports it at module level); vLLM in a separate env |

FlightGPT's loader sets `max_dist_marker_to_target=1000000`, so it keeps the
rows MGP and HETT drop at 30 m; all three drop rows past 500 teacher actions.
The released CityNav split files carry no row past 30 m, so the 30 m filter
drops nothing there.

## Cal City: the mapping, and every deviation

A released checkpoint cannot read the Cal City protocol as flown by MAPS.
`calcity_export.py` maps it onto what each checkpoint consumes:

| Cal City protocol (`configs/calcity.yaml`) | What a checkpoint gets | Why |
|---|---|---|
| One 13.4 x 10.2 km map | one 400 m block per episode, placed uniformly at random (seeded per episode) among the windows holding the start and the target shape 20 m from the edge | the checkpoints were trained on blocks of about 400 m; FlightGPT reads the whole block as one image under a 4004 px limit; HETT and MGP normalize over a 410 m canvas. A centred window would put the target at the reflection of the start |
| All 4,573 episodes | episodes whose start and target shape fit one block; the rest listed in `manifest.json` with the reason | no block holds them |
| 0.15 m/px flown ortho (EPSG:32611) | bilinear resample to 0.1 m/px, PNG + flat float32 GeoTIFF on the same grid, affine in the local frame | the checkpoints' rasters are 0.1 m/px |
| Flat world, ground 0 | `GROUND_LEVEL` 0 per block; the depth channel reads a constant 50 m | the checkpoints were trained on SensatUrban relief, and get none |
| Dataset start pose | trajectory[0], heading north | Cal City rows carry no heading |
| Start band altitude, `nadir_1008_90` sensor | 50 m above ground and each checkpoint's own crops (MGP 224/256 px plus a 500 px GSAM crop, HETT 224 px, FlightGPT a 500 px view and the whole block map) | every checkpoint flies 50 m and was trained on those crops |
| The harness's flight and time budget | each method's own budget (MGP 20 x 5 actions, HETT 20 x 10, FlightGPT 2 x 75) | the released protocol |
| CityRefer-schema catalog | objects clipped to the block (the largest part of a multi-part clip), `tags` dropped, the target added as one unnamed object | upstream's `CityReferObject` fields; unnamed so it is never a landmark |
| No human marker | marker = target; the teacher path is a straight line to it | the loader needs a marker; only HETT's SPL denominator reads the teacher path |
| CityNav's processed descriptions | an LLM parse (`describe_instructions.py`), landmarks restricted to the block's names | `LandmarkMap` snaps every query to the nearest name, and fails on a block with no landmarks |
| FlightGPT's GSAM map cache | every lookup reads empty (`--zero-gsam-cache`) | no cache exists for Cal City; FlightGPT draws only landmarks and the trajectory into the image it sends, so the GSAM layers never reach the model |
| Episodes in one split | all exported episodes as `test_unseen`; MGP's other splits load empty (`--splits`), HETT builds only `test_unseen` | the evaluators iterate fixed split names |
| Scoring: the shape, as drawn | the same, by `outcome_rows` | so the number is on the Cal City contract |

Each shard (50 blocks by default) is a separate run root: every evaluator
loads all of its rasters into memory at start, about 110 MB a block for MGP
and HETT and twice that for FlightGPT.

## Setup checks

Setup unpacks the HETT Dropbox folder, and any archive nested in it, and
finds `best_val_unseen`, `yolo_v3.cfg`, `best.pt` and the refined
`citynav_test_unseen.json` by name, stopping if any is missing. It tolerates
a failure of MGP's `download_weights.sh` (its last step fetches VLN-CE
embeddings MGP does not read) and then checks the three weights MGP reads.
It pulls the Hugging Face BERT and FlightGPT weights in the setup stage.
