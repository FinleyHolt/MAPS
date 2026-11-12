# benchmarks/citynav

This package runs the CityNav benchmark (Lee et al.,
[arXiv:2406.14240](https://arxiv.org/abs/2406.14240)) as MAPS missions. Each
episode is one mission conversation, replayed as upstream's six discrete
actions. The replay's final pose is scored by the upstream rule: success
within 20 m of the target. The package also holds the shared engine that the
Cal City bench and the flight harness hydrate a mission through, and the run
readers in [`analysis/`](analysis/README.md).

## Prepare the data

The episodes, the CityRefer catalog, and the SensatUrban point clouds come
from upstream; this repository redistributes none of them.

1. Request SensatUrban's `data_release.zip` from its authors'
   [form](https://github.com/QingyongHu/SensatUrban). The terms are
   CC BY-NC-SA, non-commercial research use, and no redistribution.

2. Run the setup script:

   ```sh
   export CITYNAV_ROOT=~/datasets/citynav-arxiv
   benchmarks/citynav/data_prep/setup.sh --sensaturban <data_release.zip>
   ```

For the steps the script runs and what each writes, see
[`docs/data.md`](../../docs/data.md#citynav).

The following variables locate the data. An entry point reads them once into
a bench value (`paths.citynav_bench(os.environ)`) and passes that value down.

| Variable | Default | Description |
|---|---|---|
| `SEARCH_CITYNAV_DATA`, then `CITYNAV_ROOT` | `~/datasets/citynav-arxiv` | Upstream checkout. |
| `SEARCH_CITYNAV_SPLITS` | `$CITYNAV_ROOT/data/citynav` | Episode split files. |
| `SEARCH_CITYNAV_RGBD_DIR` | `$CITYNAV_ROOT/data/rgbd_upstream` | Nadir rasters the observation crops. |
| `SEARCH_CITYNAV_TILES` | `tiles/citynav_2m_osmv3` | OSM prior bake. |

## Run the bench

To fly a split:

```sh
python -m benchmarks.citynav.bench_fullset --split train_seen --limit 1000 --out-root output/citynav
```

To fly one episode through the runner (its root is `--out`):

```sh
python -m benchmarks.citynav.run_benchmark --config configs/citynav.yaml --split train_seen --episode-id <id> --out output/citynav_one
```

`bench_fullset` runs a split at a fixed worker count, one fresh
`run_benchmark --attempt-root` child per attempt. Rerun a command to resume
its root.

| Flag | Description |
|---|---|
| `--split <split>` | Split to fly. |
| `--out-root <dir>` | Run root. |
| `--workers <n>` | Worker count. Default 6. |
| `--limit <n>` | Episodes kept after the seeded shuffle. A seed and a limit select the same episodes on every run. |
| `--sample-seed <n>` | Seed of the shuffle applied before `--limit`. Default 1234. |
| `--episode-allowlist <file>` | Keeps only the listed IDs, before the shuffle. A list that names an ID outside the split is refused. |
| `--arrivals <stream.json>` | Flies a report stream in place of the shuffle: a JSON object whose `reports` list holds `{"episode_id", "arrival_s"}` rows, each episode released at its arrival time on the session clock. |
| `--max-infra-attempts <n>` | Attempts an episode gets under infrastructure faults. Default 3. |
| `--dry-run` | Prints the selection and launches nothing. |

For the remaining flags, run
`python -m benchmarks.citynav.bench_fullset --help`. `CITYNAV_BENCH_CONFIG`
names the config (default `configs/citynav.yaml`).

The driver refuses or records faults as follows:

| Case | Result |
|---|---|
| Refused selection | Exits 2 before the out-root exists or a serve is asked. |
| Selection whose blocks lack their bake files or raster pair | Refused at startup, naming the files. |
| Second run into a root | Refused. A run holds its root for its lifetime (`.bench.lock`). |
| Exception raised by the method | A scored miss: status `ok` with its `error` and `traceback` and no metrics. |
| Infrastructure fault (a serve or segmenter outage, an image-store refusal, a runner that died) | Status `infra`. The episode flies again, up to `--max-infra-attempts` attempts, and past that is reported as persistent infra with its errors. |

### Run root

| Path | Contents |
|---|---|
| `run_manifest.json` | Run identity: commit, resolved config, split and selection, served model, tile bake hash, arm flags. |
| `episodes.jsonl` | One row per attempt, its scored `metrics` included. |
| `summary.txt` | `analysis.run_report` over the root under the driver's infra lines, refreshed after every attempt. |
| `driver.log` | Driver's own lines: each attempt's start and end, crashes, timeouts, aborts. |
| `w<id>/server.log` | Stderr of worker `<id>`'s attempt children, appended. |
| `w<id>/maps_<stamp>_<benchmark>_<slug>/` | One directory per attempt: `episode.json`, `run.log`, `conversation.jsonl`, and `flight/`. |

Read a root with [`analysis/`](analysis/README.md). Don't parse
`episodes.jsonl` by hand.

## Configs

`configs/citynav.yaml` is the control config and declares every key. Another
config names it with `base:` and holds only the keys it changes
(`configs/calcity.yaml`). The loader
(`benchmarks/run_record/config.py`) refuses unknown keys and checks every
value by type and range.

## Modules

| Module | Contents |
|---|---|
| `bench_fullset.py` | Full-split driver. |
| `run_benchmark/` | Runner: episode setup, the mission engine (`mission_engine.py`), scoring rows. |
| `adapter.py` | Dataset: split rows and the CityRefer catalog. |
| `paths.py` | `citynav_bench(environ)`: the bench as a value (`benchmarks.bench.Bench`), its roots read from the variables above. `selected_bench(environ)` picks the bench an attempt child flies. |
| `engine.py` | Episode engine a bench value carries. The Cal City bench, the flight harness, and the station fly through it. |
| `eval.py` | Scoring: NE, SR, OSR, SPL, and the shape criterion the Cal City bench uses. |
| `action_snap.py` | Six-action alphabet a flight is replayed in. |
| `bake/city.py` | OSM prior bake. |

Both benches and the flight harness share the run record in
`benchmarks/run_record/`:

| Module | Contents |
|---|---|
| `episodes.py` | Ledger and the attempt's row. |
| `episode_workdir.py` | An attempt's directory, its `run.log`, and the `conversation.jsonl` and `detections.jsonl` written as the mission runs. |
| `run_manifest.py` | Run manifest and the audited environment. |
| `config.py` | Run config and its loader. |
| `rows.py` | Which attempt is an episode's result. |
