# MAPS: Multimodal Agentic Planning and Search

MAPS is a method for aerial language-guided target search. An operator
describes a target in words ("the car parked north of the church"), and a
multimodal language model flies an aircraft to find it, reasoning over an
OpenStreetMap prior, the imagery its own flights collect, and an open-vocabulary
segmenter.

This repository holds the method, the two benchmarks it is measured on
(CityNav and Cal City), the flight harness that flew it in closed loop on a
C-12 over California City, and the readers that derive every reported number
from run records.

## The method

A search is one mission conversation with one reasoning model. The model
receives a system turn and the operator's words, and acts through exactly one
tool, `run`, whose argument is a block of ordinary Python. The block executes
in a namespace that persists for the whole mission, against a library of
calls over the world state:

| Call | What it does |
|---|---|
| `find` | Reads rows of the world state (the OSM prior plus every detection) by name or class, ordered or cut by place, side and colour. |
| `look_for` | Names up to four classes the segmenter detects on every later flight. |
| `go_to` | Flies to one thing on the map. |
| `explore` | Flies a covering pass over a thing or a set, segmenting at every stop, and returns the rows it found. |
| `look` | Returns a close crop of the sensor's pixels at a row, as a picture. |
| `show_map` | Returns the world-state map around a place, drawn as a schematic. |
| `stand` | Adopts a ground viewpoint, without flying, that later rows and pictures face. |
| `more_of` | Returns the next block of a long result. |
| `intersection` | Returns where two roads cross. |
| `ends_of` | Returns where a road ends, one row per terminus. |
| `distance` | Returns the metres between two footprints. |

A result is a set of rows with `.filter` (a cut) and `.order` (one sort
key). A bare handle on its own line shows its rows; `print()` and the block's
last value come back in the tool result, the receipt. Only `go_to` and
`explore` move the aircraft.

Perception is SAM 3.1 run over an orthographic observation: each flight's
view is a north-up orthophoto of the ground under the aircraft (a crop of a
raster on the benchmarks, fused camera frames in flight), segmented for the
classes the model armed with `look_for`. Detections join the OSM prior in one
world-state graph.

The mission ends when the model replies without a tool call. The commit is
the target of the mission's last flight.

The code lives in [`maps/`](maps/README.md). Print the served contract (the
system turn and the tool schema the model receives) with:

```sh
python -m maps.search.parse_compile.contract --sensor oryx_orx32_25mm_nadir
```

## The closed-loop flight

MAPS flew 87 closed-loop runs on a C-12 over California City, CA, from 11 to
18 September 2026, at 2,500 to 5,000 ft above ground. The reasoner ran off
board over the datalink; SAM 3.1 ran on board. The station's method half is
[`benchmarks/flight`](benchmarks/flight/README.md) in this repository.

The flown engine is this repository's engine. The served text (system turn,
tool schema and every string the model reads) is byte-identical to what the
station served on every run from 14 September on. The engine differs from
the flown one in these places:

| Difference | Effect on a flight |
|---|---|
| A block whose Python parses but fails to compile (a repeated keyword argument) comes back as a tool error the model reads. | The flown engine ended such an episode as an infrastructure failure. |
| A receipt that prints a row's type names the `maps` package. | The flown receipt named the flown package. |
| The aircraft is reached through the `Vehicle` protocol (post a leg, read poses and recorded frames). | The station spoke to the recorder and cockpit display over HTTP with the same semantics. |
| The bench ablation withholds (`maps/search/withhold.py`) exist; the flight mission refuses a run that names one. | None; a flight serves every part. |
| The Cal City bench reads a user-supplied map raster and derives the reviewed episode set; its `look` caption states the raster's own ground sample distance. | Bench only; flight captions state the live imager's. |

## Results

### Closed loop

The closed-loop measures are read from the flown sorties' own records by
[`benchmarks/flight/measures.py`](benchmarks/flight/measures.py):

| Measure | Value | Definition |
|---|---|---|
| SR, 20 m radius | 40/80 | The commit lies within 20 m of the target centroid. |
| SR, committed area | 33/80 | The committed feature's outline contains the target centroid. |
| OSR | 70/80 | The aircraft's path (GNSS track, else keyframe poses) came within one camera footprint width of the target centroid. |

```sh
python -m benchmarks.flight.measures --root <sorties>/* --truth <adjudications dir> <tasking split JSON>
```

A run the analysts' adjudication marks excluded is out of every rate, and so
is a run the serve failed under (the bench's infra rule, `rows.is_infra_abort`),
which the output names. The flight recordings and the adjudications are not
distributed.

### Bench experiments

Each experiment is a set of run roots; the readers in
[`benchmarks/citynav/analysis`](benchmarks/citynav/analysis/README.md) derive
every number from them. [Reproduce the experiments](#reproduce-the-experiments)
lists the exact commands.

| Experiment | Split and episodes | Arms | Config and flags | Reader |
|---|---|---|---|---|
| E1 | CityNav `train_seen`, 1,000 per arm (`--limit 1000`, seed 1234) | Inkling Small at effort 0.99; Qwen 3.8 Flash Next at `xhigh`; Gemma 4 26B-A4B | `configs/citynav.yaml`, `SEARCH_SAMPLING_REASONING_EFFORT` | `arm_table`, `run_report` |
| E2 | CityNav `test_unseen`, all 5,311 | Qwen 3.8 Flash Next at `low`, `medium`, `xhigh`; Inkling Small at 0.99; Gemma 4 E2B; Gemma 4 E4B | `configs/citynav.yaml`, `--split test_unseen` | `arm_table`, `run_report` |
| E3 | Cal City, the 606 reviewed episodes | Inkling Small at 0.0, 0.3, 0.6, 0.9, 0.99; Gemma 4 26B-A4B | `configs/calcity.yaml`, `--episode-allowlist` from `benchmarks.calcity.data_prep.reviewed` | `arm_table`, `figures.plot_effort_ladder` |
| E4 | CityNav `train_seen`, 1,000 per arm | Per model: the control and one arm per withhold | `configs/withhold/<part>.yaml` | `arm_table`, `ab_report` |
| E5 | E2 and E3 roots | Every arm | none beyond E2 and E3 | `figures.capability_effort_arms`, `figures.plot_capability_effort` |

## Install

MAPS needs Python 3.12 or later. With [uv](https://docs.astral.sh/uv/):

```sh
uv venv .venv --python 3.12
uv pip install --python .venv -e '.[dev]'
```

Or with pip:

```sh
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

Extras:

| Extra | For |
|---|---|
| `bake` | The CityNav OSM prior bake (`osmnx`, `geopandas`). |
| `recording` | Reading a camera recording for the as-flown frame replay (`mcap`). |
| `dev` | The test suite and linter. |

The SAM 3.1 serve runs in its own environment, because Meta's `sam3` package
requires `numpy<2`:

```sh
micromamba env create -f envs/sam3_1.yml
micromamba activate sam3_1
pip install <a checkout of Meta's SAM 3 source>
```

The checkpoint `facebook/sam3.1` is gated on Hugging Face: accept its terms,
export `HF_TOKEN`, and the serve fetches it on first load, or point
`SAM3_CHECKPOINT` at a local file.

Run the tests:

```sh
.venv/bin/pytest -q
```

## Endpoints

MAPS calls two services: an OpenAI-compatible reasoner and the SAM 3.1 serve.

Start the SAM serve from the repository root, in the `sam3_1` environment:

```sh
python -m serving.sam_server --port 8772
```

The reasoner is any OpenAI-compatible chat-completions endpoint whose
`/v1/models` lists the served model and its `max_model_len` (vLLM reports
both). The served model name selects a sampling envelope from
[`maps/llm/model_cards.py`](maps/llm/model_cards.py) (tokenizer, `top_p`,
`top_k`, the reasoning efforts it takes), and must be one of that table's
keys exactly: `inkling-small`, `qwen38-flash-next-nvfp4`,
`gemma4-26b-a4b-nvfp4` (E1 to E3), `gemma4-e4b`, `gemma4-e2b` (E2), and
`gemma4-26b-a4b-gguf` (the llama.cpp serve flown aboard); any other name is
refused at startup with the list. Every request runs in thinking mode at
temperature 1.0, and `max_tokens` is the output reserve of the discovered
window.

| Variable | Read by | Meaning |
|---|---|---|
| `SEARCH_REASONER_URL` | client | The reasoner's base URL, one endpoint. Default `http://127.0.0.1:8001/v1`. |
| `SEARCH_REASONER_API_KEY` | client | Bearer token for the reasoner. |
| `SEARCH_REASONER_TIMEOUT_S` | client | Per-call transport timeout, seconds. |
| `SEARCH_ENDPOINT_SPEC_OVERRIDE` | client | `<model_id>:<max_model_len>` for an endpoint whose `/v1/models` reports no window. |
| `SEARCH_LLM_ROUTING_KEY_HEADER` | client | The header name the per-episode routing key rides (default `x-smg-routing-key`). |
| `SEARCH_SAMPLING_REASONING_EFFORT` | client | The arm's reasoning effort, sent verbatim as the request's `reasoning_effort` string (`0.99`, `xhigh`); discovery refuses a value the served model's card does not take. Unset sends none. |
| `SEARCH_SAMPLING_TEMPERATURE`, `SEARCH_SAMPLING_PRESENCE_PENALTY` | client | Replace the envelope's temperature, or add a presence penalty. |
| `SEARCH_COMPACT_TOKENIZER_NAME` | client | The HF tokenizer repo, replacing the model card's. |
| `SAM_SERVER_URL` | client | The SAM serve's URL, or a comma list of replicas. Default `http://127.0.0.1:8772` when unset; a set but blank value is refused. |
| `SAM_VIDEO_SERVER_URL` | client | A separate URL for video calls. Default `SAM_SERVER_URL`. |
| `SAM_PORT`, `SAM_MODEL_ID`, `SAM_DEVICE`, `SAM_DTYPE` | serve | Port (8772), checkpoint id (`facebook/sam3.1`), device, dtype (`bfloat16`). |
| `SAM3_CHECKPOINT` | serve | A local checkpoint file. |
| `SAM_MAX_INFLIGHT`, `SAM_INFLIGHT_WAIT_S`, `SAM_GPU_CHUNK`, `SAM_PREWARM`, `SAM_HEALTH_MIN_FREE_GB` | serve | Admission, batching and health limits (`serving/sam_server.py`). |

## Data

### CityNav

CityNav's episodes, the CityRefer catalog and the SensatUrban point clouds
come from their upstream sources; this repository redistributes none of them.

```sh
export CITYNAV_ROOT=~/datasets/citynav-arxiv
benchmarks/citynav/data_prep/fetch_data.sh             # episodes and catalog; prints the SensatUrban steps
benchmarks/citynav/data_prep/rasterize_citynav_native.sh   # nadir rasters from the block point clouds
PY=.venv/bin/python benchmarks/citynav/data_prep/bake_tiles.sh   # the OSM prior, needs the bake extra
```

The rasterizer runs upstream's PDAL pipeline in an environment with `pdal`,
`rasterio` and `pillow` (`RASTERIZE_PYTHON` names its interpreter). The bake
writes `tiles/citynav_2m_osmv3`. See
[`benchmarks/citynav`](benchmarks/citynav/README.md) for the variables that
relocate each part.

### Cal City

The Cal City corpus (`benchmarks/calcity/data/calcity.json`, 1,023 episodes)
and the reviewed target shapes
(`benchmarks/calcity/data_prep/authoring/ground_truth.json`) ship with the
repository. You supply two inputs:

* The C-12 v6 map raster of the block, `<map>_rgb.tif`.
* A world bundle of OpenStreetMap layers over the block (`manifest.json` and
  `osm/<layer>.geojson` in WGS84), named by `CALCITY_BUNDLE`.

Then build the catalog, the raster pair and the tile bake:

```sh
export CALCITY_BUNDLE=<bundle dir>
python -m benchmarks.calcity.data_prep.build_objects_json
python -m benchmarks.calcity.data_prep.build_rgbd_ortho --raster <map>_rgb.tif
python -m benchmarks.calcity.data_prep.bake_tiles
```

The flight harness also reads the block's bare-earth surface, as the flown
bake carried it: `calcity_dem.tif` beside `calcity_meta.json` under
`SEARCH_CALCITY_TILES`, heights in metres NAVD88 on 10 m cells in EPSG:32611
over the bake's bounds. It is public USGS 3DEP data (the 1/3 arc-second DEM,
from The National Map), resampled onto the bake's frame, for example:

```sh
gdalwarp -t_srs EPSG:32611 -tr 10 10 -r bilinear -te <origin_e> <origin_n> <origin_e + x_max> <origin_n + y_max> \
  USGS_13_*.tif "$SEARCH_CALCITY_TILES/calcity_dem.tif"
```

`origin_e` and `origin_n` are the bake's `georef` in `calcity_meta.json`,
`x_max` and `y_max` its `bounds`. Without the file the flight places every
frame over flat ground at the block's floor (`tile_georef.FLOOR_MSL_M`).

See [`benchmarks/calcity`](benchmarks/calcity/README.md).

## Reproduce the experiments

Each command writes one run root. Name a model by serving it at
`SEARCH_REASONER_URL`; the run manifest records the served model the
endpoint reports. A rerun of the same command resumes the root.

E1, one model at a time:

```sh
SEARCH_SAMPLING_REASONING_EFFORT=0.99 \
  python -m benchmarks.citynav.bench_fullset --split train_seen --limit 1000 --sample-seed 1234 \
  --out-root output/e1/inkling-small
SEARCH_SAMPLING_REASONING_EFFORT=xhigh \
  python -m benchmarks.citynav.bench_fullset --split train_seen --limit 1000 --sample-seed 1234 \
  --out-root output/e1/qwen38-flash-next
python -m benchmarks.citynav.bench_fullset --split train_seen --limit 1000 --sample-seed 1234 \
  --out-root output/e1/gemma4-26b-a4b
```

E2, each arm over the whole split:

```sh
SEARCH_SAMPLING_REASONING_EFFORT=xhigh \
  python -m benchmarks.citynav.bench_fullset --split test_unseen --out-root output/e2/qwen38-flash-next-xhigh
```

The other arms change the effort (`low`, `medium`, or `0.99` with Inkling
Small served), or serve Gemma 4 E2B or E4B with no effort set.

E3, the reviewed Cal City episodes per effort rung:

```sh
mkdir -p output && python -m benchmarks.calcity.data_prep.reviewed > output/reviewed_episodes.txt
for e in 0.0 0.3 0.6 0.9 0.99; do
  SEARCH_SAMPLING_REASONING_EFFORT=$e SEARCH_CALCITY_RGBD_DIR=<raster dir> \
    python -m benchmarks.calcity.bench_fullset --episode-allowlist output/reviewed_episodes.txt \
    --out-root output/e3/inkling-small-e$e
done
SEARCH_CALCITY_RGBD_DIR=<raster dir> python -m benchmarks.calcity.bench_fullset \
  --episode-allowlist output/reviewed_episodes.txt --out-root output/e3/gemma4-26b-a4b
```

E4, the control and one root per withheld part, per model:

```sh
for part in prior look_for look images mission_map system_turn; do
  CITYNAV_BENCH_CONFIG=configs/withhold/$part.yaml \
    python -m benchmarks.citynav.bench_fullset --split train_seen --limit 1000 --sample-seed 1234 \
    --out-root output/e4/<model>/$part
done
```

The control is the model's E1 root.

Read the results:

```sh
python -m benchmarks.citynav.analysis.arm_table output/e2/*            # SR, OSR, NE per arm
python -m benchmarks.citynav.analysis.run_report output/e1/inkling-small   # one root in full
python -m benchmarks.citynav.analysis.ab_report output/e1/<model> output/e4/<model>/look --allow-provenance-mismatch
python -m figures.plot_effort_ladder output/e3/inkling-small-e*
python -m figures.plot_capability_effort --split test_unseen output/e2/*
python -m figures.plot_capability_effort --split calcity --cost total output/e3/*
```

Figures land in `output/figures/` as light and dark SVGs. Every claimed
number and figure maps to its reader in
[`benchmarks/citynav/analysis/README.md`](benchmarks/citynav/analysis/README.md).

## Fly the method

[`benchmarks/flight`](benchmarks/flight/README.md) holds the flight harness:

* `Mission` (`benchmarks/flight/mission.py`) hydrates the engine as the bench
  does and holds the conversation open across operator taskings. A flight is
  a leg the aircraft flies, arrived when the recording holds the area it was
  flown for.
* `Vehicle` (`benchmarks/flight/vehicle.py`) is the boundary to the aircraft:
  `post(leg)`, `pose(t)`, `index(since)`, `frame(seq)`, `now()`, `sleep(s)`.
* `LiveSource` (`benchmarks/flight/live_source.py`) turns the recorded frames
  into the observation the model reads: coverage, per-frame registration, the
  pose-graph corrector, compositing and the mosaic fit.

The twin is a kinematic `Vehicle` over a raster pair, so you can fly a Cal
City mission end to end without an aircraft:

```sh
python -m benchmarks.flight.twin --config configs/calcity.yaml --episode 'calcity::6#6' --ortho <raster dir>
```

The twin, the live source and the frame registration stand each frame on
the bake's `calcity_dem.tif` where it is present (see [Cal City](#cal-city)).

## Replay a flown run

`python -m benchmarks.flight.replay` flies a recorded run again on this
repository's harness, at three tiers:

| Command | Answers each flight with | Shows |
|---|---|---|
| `replay <run>` | The recorded observations that hold the leg's area (`--bank week`: every run under the sorties root), with your reasoner and segmenter. | How the method decides on the ground the aircraft actually saw. |
| `replay <run> --forced` | The recorded reasoner responses, observations and detections. Every request's text is diffed against the recording; differing pictures are counted (`pictures_differ`). | Whether this harness sends the flown request text turn for turn. |
| `replay <run> --frames <sensors.mcap> --boresight <chain.json> --raster <raster dir>` | Observations re-fused from the camera recording through the live chain. | The flight-side observation chain on the real frames. |

The station registered frames against NAIP 2024 imagery: the `calcity.png`
of the v5 Cal City data release beside the flown bake. Pass that raster's
directory as `--raster` to replay as flown.

## Repository layout

| Path | Contents |
|---|---|
| [`maps/`](maps/README.md) | The method: mission loop, Python surface, world state, perception client, LLM transport. |
| [`benchmarks/citynav/`](benchmarks/citynav/README.md) | The CityNav bench: episode runner, scoring, prior bake, data preparation. |
| [`benchmarks/citynav/analysis/`](benchmarks/citynav/analysis/README.md) | The run readers every number is derived with. |
| [`benchmarks/calcity/`](benchmarks/calcity/README.md) | The Cal City bench: corpus, reviewed set, raster and bake build. |
| [`benchmarks/flight/`](benchmarks/flight/README.md) | The flight harness, the twin, as-flown replay and the closed-loop measures. |
| `figures/` | The effort-ladder and capability-effort figures. |
| `serving/` | The SAM 3.1 serve. |
| `configs/` | Bench configs: the CityNav control, Cal City, and the withhold arms. |
| `envs/` | The SAM 3.1 serve environment. |

## Citation

```bibtex
@software{holt2026maps,
  title   = {MAPS},
  author  = {Holt, Finley R.},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/FinleyHolt/MAPS}
}
```

MAPS evaluates on CityNav (Lee et al.,
[arXiv:2406.14240](https://arxiv.org/abs/2406.14240)), built on SensatUrban and
CityRefer; cite them as well. [`CITATION.cff`](CITATION.cff) carries the same
record.

## License

MIT; see [`LICENSE`](LICENSE). Third-party code and data are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
