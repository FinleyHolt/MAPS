# benchmarks/citynav

The CityNav benchmark (Lee et al., [arXiv:2406.14240](https://arxiv.org/abs/2406.14240))
run as MAPS missions. Each episode is one mission conversation, replayed as
upstream's six discrete actions; the replay's final pose is scored against the
episode's target by the upstream rule: success within 20 m of the target. This package also holds the shared
engine the Cal City bench and the flight harness hydrate a mission through,
and the run readers in [`analysis/`](analysis/README.md).

## Prepare the data

The episodes, the CityRefer catalog and the SensatUrban point clouds come from
upstream; nothing is redistributed here.

```sh
export CITYNAV_ROOT=~/datasets/citynav-arxiv
benchmarks/citynav/data_prep/fetch_data.sh
benchmarks/citynav/data_prep/rasterize_citynav_native.sh
PY=.venv/bin/python benchmarks/citynav/data_prep/bake_tiles.sh
```

| Step | Writes | Notes |
|---|---|---|
| `fetch_data.sh` | `$CITYNAV_ROOT/data/citynav/*.json`, `$CITYNAV_ROOT/data/cityrefer/objects.json` | Clones `water-cookie/citynav-arxiv` and runs its downloader. The 34 SensatUrban block PLYs are placed by hand under `$CITYNAV_ROOT/raw_ply/`; the script prints where to get them. |
| `rasterize_citynav_native.sh [--blocks <block>...]` | `$CITYNAV_ROOT/data/rgbd_upstream/<block>.{png,tif}` | Runs upstream's `rasterize.py` (vendored as `data_prep/_vendor_citynav_rasterize.py`) in an environment with `pdal`, `rasterio` and `pillow`; `RASTERIZE_PYTHON` names its interpreter. |
| `bake_tiles.sh` | `tiles/citynav_2m_osmv3/` | Bakes every block with a raster under `SEARCH_CITYNAV_RGBD_DIR` from the city's shipped OpenStreetMap cache, `tiles/osm_cache/<city>_osm_local.geojson` (already in CityNav local XY), on 2 m cells over the raster's extent. |

| Variable | Default | Names |
|---|---|---|
| `SEARCH_CITYNAV_DATA`, then `CITYNAV_ROOT` | `~/datasets/citynav-arxiv` | The upstream checkout. |
| `SEARCH_CITYNAV_SPLITS` | `$CITYNAV_ROOT/data/citynav` | The episode split files. |
| `SEARCH_CITYNAV_RGBD_DIR` | `$CITYNAV_ROOT/data/rgbd_upstream` | The nadir rasters the observation crops. |
| `SEARCH_CITYNAV_TILES` | `tiles/citynav_2m_osmv3` | The OSM prior bake. |

## Run the bench

```sh
python -m benchmarks.citynav.bench_fullset --split train_seen --limit 1000 --out-root output/citynav
```

`bench_fullset` runs a split at a fixed worker count (`--workers`, default 6),
one fresh `run_benchmark --attempt-root` child per attempt. The split's
episodes (after `--episode-allowlist`) are shuffled by `--sample-seed`
(default 1234) before `--limit` applies, so a seed and a limit select the
same episodes on every run.
`--episode-allowlist` keeps only the listed ids and refuses a list naming
an id outside the split. A refused selection exits 2 before the out-root
exists or a serve is asked. Rerunning a command resumes its root. `--dry-run` prints the selection and launches nothing.
`--arrivals <stream.json>` flies a capacity session in place of the shuffle
(`benchmarks/capacity/README.md`).
`CITYNAV_BENCH_CONFIG` names the config (default `configs/citynav.yaml`).

A run holds its root for its lifetime (`.bench.lock`); a second run into it
is refused. Startup refuses a selection whose blocks lack their bake files or
raster pair, naming the files. An exception the method raises lands as a
scored miss: status `ok` with its `error` and `traceback` and no metrics.
Only an infrastructure fault (a serve or segmenter outage, an image-store
refusal, a runner that died) lands `infra`; such an episode flies again, up
to `--max-infra-attempts` (3) attempts, and past that is reported as
persistent infra with its errors.

A run root holds:

| Path | Contents |
|---|---|
| `run_manifest.json` | The run's identity: commit, resolved config, split and selection, served model, tile bake hash, arm flags. |
| `episodes.jsonl` | One row per attempt, its scored `metrics` included. |
| `summary.txt` | `analysis.run_report` over the root under the driver's infra lines, refreshed after every attempt. |
| `driver.log` | The driver's own lines: each attempt's start and end, crashes, timeouts, aborts. |
| `w<id>/server.log` | Worker `<id>`'s attempt children's stderr, appended. |
| `w<id>/maps_<stamp>_<benchmark>_<slug>/` | One directory per attempt: `episode.json`, `run.log`, `conversation.jsonl` and `flight/`. |

Read a root with [`analysis/`](analysis/README.md); never parse
`episodes.jsonl` by hand.

One episode, straight through the runner (its root is `--out`):

```sh
python -m benchmarks.citynav.run_benchmark --config configs/citynav.yaml --split train_seen --episode-id <id> --out output/citynav_one
```

## Configs

`configs/citynav.yaml` is the control config and declares every key. Another
config names it with `base:` and holds only the keys it changes
(`configs/calcity.yaml`, `configs/withhold/*.yaml`). Unknown keys are refused
and every value is checked by type and range at load
(`run_benchmark/config.py`).

## Modules

| Module | Role |
|---|---|
| `bench_fullset.py` | The full-split driver. |
| `run_benchmark/` | The runner: config, episode setup, the mission engine (`mission_engine.py`), scoring rows. |
| `adapter.py`, `episodes.py`, `paths.py` | The dataset, the ledger, the file layout. |
| `episode_workdir.py` | An attempt's directory, its `run.log`, and the `conversation.jsonl` and `detections.jsonl` written as the mission runs. |
| `eval.py` | Scoring: NE, SR, OSR, SPL, and the shape criterion the Cal City bench uses. |
| `action_snap.py` | The six-action alphabet a flight is replayed in. |
| `bake/city.py` | The OSM prior bake. |
| `run_manifest.py` | The run manifest and the audited environment. |
