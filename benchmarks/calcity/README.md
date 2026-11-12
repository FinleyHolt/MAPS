# benchmarks/calcity

This package is the Cal City benchmark: target search over California City,
CA, the area the C-12 flew. It runs the same engine and flight contract as
[CityNav](../citynav/README.md) over a 2D world: the block's map raster on a
flat height plane, an OpenStreetMap prior, and an authored corpus. An episode
succeeds when its commit point lies inside the target's own annotated
footprint (the shape criterion), the same test as the flight measures.

## Data

The corpus ships in this package. The map raster and the OSM bundle don't
ship in this repository.

| Item | Path | Source |
|---|---|---|
| Corpus, 4,573 episodes | `data/calcity.json` | Ships. |
| Positions of the reviewed targets | `data_prep/authoring/ground_truth.json` | Ships. |
| Map raster of the block | `<map>_rgb.tif` | You supply it: the C-12's map, or a public orthophoto on the block's frame. For the frame, see [`docs/data.md`](../../docs/data.md#cal-city). |
| World bundle of OpenStreetMap layers | `osm/<layer>.geojson` in WGS84 for each of the eight `bundle.OSM_LAYERS` | You supply it, covering `aoi.BUNDLE_BBOX_WGS84`. |
| Bare-earth surface of the block, read by the flight harness | `calcity_dem.tif` under `SEARCH_CALCITY_TILES` | You supply it from public USGS 3DEP data. See [`docs/data.md`](../../docs/data.md#flight-harness-dem). |

## Build the bench

1. Name the world bundle:

   ```sh
   export CALCITY_BUNDLE=<bundle dir>
   ```

2. Build the CityRefer-schema catalog, `<root>/data/cityrefer/objects.json`:

   ```sh
   python -m benchmarks.calcity.data_prep.build_objects_json
   ```

3. Build the raster pair under `<root>/data/rgbd_upstream`: the raster as
   given, `<map>_rgb.tif`, and a constant height plane on its grid,
   `<map>.tif`. The raster must be the block's frame at its own cell:

   ```sh
   python -m benchmarks.calcity.data_prep.build_rgbd_ortho --raster <map>_rgb.tif
   ```

4. Bake the OSM prior and `calcity_meta.json` under `SEARCH_CALCITY_TILES`:

   ```sh
   python -m benchmarks.calcity.data_prep.bake_tiles
   ```

`<root>` is the build commands' `--root`. It defaults to the bench's build
tree (`SEARCH_CALCITY_DATA`, then `CALCITY_ROOT`). `bake_tiles --tiles-dir`
defaults to `SEARCH_CALCITY_TILES`.

| Variable | Default | Description |
|---|---|---|
| `SEARCH_CALCITY_DATA`, then `CALCITY_ROOT` | `~/datasets/calcity` | Build tree. |
| `SEARCH_CALCITY_RGBD_DIR` | `<root>/data/rgbd_upstream` | Raster pair the observation crops. A second raster of the block goes in a second directory. |
| `SEARCH_CALCITY_TILES` | `benchmarks/calcity/tiles` | OSM prior bake. |
| `SEARCH_CALCITY_SPLITS` | `benchmarks/calcity/data` | Corpus. |
| `CALCITY_BUNDLE` | `<root>/bundle` | World bundle. |

## Reviewed episodes

The reviewed set is derived: a corpus row is reviewed when its first target
position lies within 0.2 m of a position that `ground_truth.json` lists on
its own map. To write the set's episode IDs, one per line:

```sh
mkdir -p output && python -m benchmarks.calcity.data_prep.reviewed > output/reviewed_episodes.txt
```

To count the set, run `wc -l output/reviewed_episodes.txt`.

A row's `intel_quality` is the class of its graded candidate count.
`analysis.run_report` breaks SR out by it.

| `intel_quality` | Candidates |
|---|---|
| G | One |
| M | Two to five |
| P | Six or more |
| None | A hand-written row whose count was never graded |

## Run the bench

To fly the reviewed episodes:

```sh
python -m benchmarks.calcity.bench_fullset --episode-allowlist output/reviewed_episodes.txt --out-root output/calcity
```

To fly one episode:

```sh
python -m benchmarks.calcity.run_benchmark --config configs/calcity.yaml --episode-id 'calcity::6#6' --out output/calcity_one
```

`bench_fullset` takes the [CityNav driver's flags](../citynav/README.md#run-the-bench)
and reads `configs/calcity.yaml`: the C-12's Oryx camera at 3,000 ft above
ground from a seeded uniform start. Read a root with
[`benchmarks/citynav/analysis`](../citynav/analysis/README.md).

Each launcher builds its bench once, from the process environment, and passes
it down. Nothing selects a bench for the process, so one process can hold a
CityNav bench and a Cal City bench side by side. An attempt child of the
driver learns its bench from the environment the bench carries
(`SEARCH_BENCHMARK_NAME` and the roots under the runner's `SEARCH_CITYNAV_*`
names), which every run manifest also records.

## Modules

| Module | Contents |
|---|---|
| `paths.py` | `calcity_bench(environ)`: the bench as a value (`benchmarks.bench.Bench`), its roots read from the `SEARCH_CALCITY_*` variables of the mapping it is given. It flies the CityNav engine. |
| `bench_fullset.py`, `run_benchmark.py` | CityNav driver and runner on the Cal City bench. |
| `data_prep/aoi.py` | Area, its local metric frame, and the flat ground plane. |
| `data_prep/bundle.py` | Reader of the world bundle. |
| `data_prep/build_objects_json.py`, `data_prep/build_rgbd_ortho.py`, `data_prep/bake_tiles.py` | Build steps. |
| `data_prep/reviewed.py` | Reviewed episode set. |
