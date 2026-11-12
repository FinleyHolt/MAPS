# MAPS: Multimodal Agentic Planning and Search

MAPS finds a target on the ground from a description in words. A multimodal
language model flies the aircraft by writing Python against a small library
over a structured world state: an OpenStreetMap prior joined with the
detections of an open-vocabulary segmenter (SAM 3.1).

![A MAPS search flown on a C-12 over California City: the description, the flown track, and each call the model made, at the place it made it.](docs/figures/overview.webp)

*A search flown in closed loop on a C-12. Bottom: the track over California
City, with each call the model wrote at the place it wrote it.*

This repository contains the method, two benchmarks (the public CityNav, and
Cal City), the flight harness, a web console for watching and reviewing
searches, and the readers that derive every reported number from run records.

## Quickstart: review recorded searches

Requires Python 3.12 or later. No model, GPU, or dataset.

1. Install MAPS with the console:

   ```sh
   git clone https://github.com/FinleyHolt/MAPS && cd MAPS
   python -m venv .venv && source .venv/bin/activate
   pip install -e '.[station]'
   ```

2. Download `MAPS-example-runs.zip` from the
   [Releases page](https://github.com/FinleyHolt/MAPS/releases) and unzip it.
   It holds ten Cal City searches: each one's conversation, every picture the
   model was sent, and its score.

3. Open the console on the unzipped folder:

   ```sh
   SEARCH_CALCITY_TILES=<folder>/bake python -m station.serve --bench calcity \
       --out <folder>/runs --basemap <folder>/basemap.jpg
   ```

4. Go to `http://127.0.0.1:8788`. On a mission, click the run count, then the
   run.

The review shows the model's reasoning and code turn by turn, the pictures it
was sent, its path on the map, and its score. For the console's other modes,
see [`station/README.md`](station/README.md).

## Run the CityNav benchmark

CityNav ([arXiv:2406.14240](https://arxiv.org/abs/2406.14240)) is the public
benchmark.

### Prerequisites

| Requirement | Details |
|---|---|
| Reasoner | An OpenAI-compatible endpoint, self-hosted or hosted, that serves a model with a card in [`maps/llm/model_cards.py`](maps/llm/model_cards.py). |
| Segmenter | SAM 3.1 on one GPU. The checkpoint `facebook/sam3.1` is gated on Hugging Face. Meta's SAM License covers the checkpoint and the `sam3` package, and bars use "for any activities subject to the International Traffic in Arms Regulations (ITAR)" ([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md#used-and-not-redistributed)). |
| CityNav annotations | Downloaded by the setup script. |
| SensatUrban point clouds | Requested from the authors' [form](https://github.com/QingyongHu/SensatUrban). The terms are CC BY-NC-SA, non-commercial research use, and no redistribution, so this repository ships no raster or picture made from them. |

### Steps

1. Prepare the data. The script downloads the annotations, rasterizes the
   point clouds, and bakes the OpenStreetMap prior:

   ```sh
   export CITYNAV_ROOT=~/datasets/citynav-arxiv
   benchmarks/citynav/data_prep/setup.sh --sensaturban <data_release.zip>
   ```

2. Start the segmenter in its own environment, from the repository root:

   ```sh
   micromamba env create -f envs/sam3_1.yml && micromamba activate sam3_1
   pip install git+https://github.com/facebookresearch/sam3@2814fa619404a722d03e9a012e083e4f293a4e53
   HF_TOKEN=<your token> python -m serving.sam_server --port 8772
   ```

3. Point MAPS at the reasoner and check it:

   ```sh
   export SEARCH_REASONER_URL=http://<host>:<port>/v1
   export SEARCH_REASONER_API_KEY=<key>   # if the endpoint requires one
   python -m maps.mission.endpoint_preflight
   ```

4. Fly a split and read the result:

   ```sh
   python -m benchmarks.citynav.bench_fullset --split test_unseen --limit 100 --out-root output/citynav
   python -m benchmarks.citynav.analysis.run_report output/citynav
   ```

   `run_report` prints SR, NE, SPL, and OSR. Rerun the first command to resume
   an interrupted run.

5. Review the searches, or fly one from the console:

   ```sh
   python -m station.serve --bench citynav --out output/citynav
   ```

For data details, see [`docs/data.md`](docs/data.md). For endpoints and their
environment variables, see [`docs/endpoints.md`](docs/endpoints.md). For the
run root's layout and the driver's flags, see
[`benchmarks/citynav`](benchmarks/citynav/README.md).

## How it works

A search is one conversation with one model. The model has one tool, `run`,
whose argument is a block of Python. Each block reads or extends the
world-state graph, and its printed output returns to the model, which then
writes the next block. The search ends when the model replies without a tool
call; the answer is the target of its last flight.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/how-it-works-dark.svg">
  <img alt="The model writes a block of Python, the block reads or extends the world-state graph, and the result returns to the model for its next step." src="docs/figures/how-it-works-light.svg">
</picture>

*One search of 13 steps. The model finds a street in the map, flies a pass
over it, looks at the cars the pass detected, and flies to the one it picks.*

The world-state graph holds everything the search knows as rows: map features
from OpenStreetMap, and the objects that the segmenter detects on each flight.
The model's Python calls this library:

| Call | Description |
|---|---|
| `find` | Reads rows of the world state by name or class, ordered or cut by place, side, and colour. |
| `look_for` | Names up to four classes for the segmenter to detect on later flights. With `at=<place>`, also segments that place for them: a stop already flown there at once (on the bench), else on the next flight. |
| `go_to` | Flies to one row. |
| `explore` | Flies a covering pass over a row or a set, and returns the rows it detected. |
| `look` | Returns a close crop of the sensor's pixels at a row. |
| `show_map` | Returns a schematic of the world state around a place. |
| `stand` | Sets a ground viewpoint, without flying, that later rows and pictures face. |
| `intersection`, `ends_of`, `distance` | Road crossings, road ends, and metres between footprints. |
| `more_of` | Returns the next block of a long result. |

Only `go_to` and `explore` move the aircraft.

The `run` tool executes the model's Python in the process that runs the
mission, with that process's privileges. To fly a model you don't trust, run
MAPS inside an isolated environment, such as a container or a virtual machine.

To print the exact system turn and tool schema that the model receives:

```sh
python -m maps.mission.contract --sensor oryx_orx32_25mm_nadir
```

For the code behind each part, see [`maps/README.md`](maps/README.md) and
[`docs/architecture.md`](docs/architecture.md).

## Reproduce the paper

- The main branch reproduces the CityNav and Cal City runs of the full
  method. For the commands and readers, see
  [`docs/experiments.md`](docs/experiments.md).
- The ablation arms, the comparison against the baselines, and the as-flown
  replay reproduce from the tag
  [`paper-v1`](https://github.com/FinleyHolt/MAPS/tree/paper-v1)
  (`git checkout paper-v1`).
- The closed-loop flight recordings aren't distributed.

## Use your own aircraft

The flight harness reaches an aircraft through one protocol, `Vehicle`
([`benchmarks/flight/vehicle.py`](benchmarks/flight/vehicle.py)): post a leg,
read poses, and read recorded frames. Implement it, and the mission, the
observation chain, and the console run over your aircraft. The kinematic twin
flies a Cal City mission through the same harness:

```sh
python -m benchmarks.flight.twin --config configs/calcity.yaml --episode 'calcity::6#6' --ortho <root>/data/rgbd_upstream
```

See [`docs/flight.md`](docs/flight.md).

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
